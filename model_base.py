import keras.layers as layers
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine import Layer
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Last Update : 2018.07.13
# Produce by Kim You Jin, Lee Jung Hoon 
# 영화 summary 사용해서 rotten tomato score 예측 하는 파일.


# 데이터 불러오기
def load_data(train_test_dir, train_dir, test_dir, val_dir):
    with open(train_test_dir, 'rb') as rotten_file:
        new_df = pickle.load(rotten_file)
        new_df = pd.DataFrame(new_df)

    with open(train_dir, 'rb') as rotten_file:
        train = pickle.load(rotten_file)
        train = pd.DataFrame(train)

    with open(test_dir, 'rb') as rotten_file:
        test = pickle.load(rotten_file)
        test = pd.DataFrame(test)

    with open(val_dir, 'rb') as rotten_file:
        val = pickle.load(rotten_file)
        val = pd.DataFrame(val)

    train, test, val = train.reset_index(), test.reset_index(), val.reset_index()
    train_y, test_y, val_y = train["label"], test["label"], val["label"]
    train_y, test_y, val_y = train_y.tolist(), test_y.tolist(), val_y.tolist()

    train_y, test_y, val_y = np.array(train_y), np.array(test_y), np.array(val_y)
    train_y, test_y, val_y = np.matrix(train_y).reshape(len(train_y), 1), np.matrix(test_y).reshape(len(test_y), 1), np.matrix(val_y).reshape(len(val_y), 1)

    DataSet = {'train': train, 'train_y': train_y, 'test': test, 'test_y': test_y, 'val': val, 'val_y': val_y, 'total_data': new_df}

    return DataSet


# Summary에서 감정 점수 뽑아내기
def sentiment_processing(DataSet, senti_shape):
    split_article_content = []

    for element in DataSet['plot']:
        split_article_content.append(re.split("(?<=[.!?])\s+", element))

    sid = sia()
    senti_list = []

    for i in range(len(split_article_content)):
        words = split_article_content[i]
        sentiment_com, sentiment_pos, sentiment_neg, sentiment_neu = [], [], [], []
        script = []

        for word in words:
            ss = sid.polarity_scores(word)
            sentiment_com.append(ss['compound'])
            sentiment_pos.append(ss['pos'])
            sentiment_neg.append(ss['neg'])
            sentiment_neu.append(ss['neu'])
            script.append(word)

        percentile_list = pd.DataFrame(
            {
                'sentiment_sc': sentiment_com,
                'sentiment_pos': sentiment_pos,
                'sentiment_neg': sentiment_neg,
                'sentiment_neu': sentiment_neu,
                'script': script
            })
        senti_list.append(percentile_list)

    sentiment_sc__ = []
    for i in range(len(senti_list)):
        temp = []

        for a in range(len(senti_list[i]["sentiment_sc"])):
            temp.append(senti_list[i]["sentiment_sc"][a])

        sentiment_sc__.append(temp)

    def pad(l, content, width):
        zero_ = [content] * (width - len(l))
        zero_.extend(l)
        return zero_

    padding_ = []
    for i in range(len(senti_list)):
        padding_.append(pad(sentiment_sc__[i], 0, senti_shape))


    for i in range(0, len(padding_)):
        if len(padding_[i]) != senti_shape:
            print(len(padding_[i]))

    second_x = np.array(padding_)
    sentiment = second_x.reshape(len(padding_), senti_shape)

    Sentiment_DataSet = {'sentiment': sentiment}

    return Sentiment_DataSet


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


# 첫번째 모델
# ELMo 256 차원 단일 모델
def build_model_elmo():
    input_text = layers.Input(shape=(1,), dtype="string")

    embedding = ElmoEmbeddingLayer()(input_text)
    dense_elmo = layers.Dense(256, activation='relu')(embedding)

    pred = layers.Dense(1, activation='sigmoid')(dense_elmo)

    model = Model(inputs=[input_text], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# 두번째 모델
# ELMO 256차원 + Bi_LSTM 128차원
def build_model_lstm(senti_shape):
    input_elmo = layers.Input(shape=(1,), dtype="string")
    input_sentiment = Input(shape=(senti_shape, 1))

    embedding = ElmoEmbeddingLayer()(input_elmo)
    dense_elmo = layers.Dense(256, activation='relu')(embedding)

    bi_lstm_sentiment1 = Bidirectional(LSTM(units=128, kernel_initializer='glorot_normal', return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(input_sentiment)
    bi_lstm_sentiment2 = Bidirectional(LSTM(units=128, kernel_initializer='glorot_normal', return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(bi_lstm_sentiment1)

    adding = layers.add([bi_lstm_sentiment1, bi_lstm_sentiment2])  # residual connection to the first biLSTM
    dense_sentiemnt = layers.Flatten()(adding)

    merge = concatenate([dense_elmo, dense_sentiemnt])
    merge = layers.Dense(128, kernel_initializer='glorot_normal', activation='relu')(merge)

    output = Dense(1, activation='sigmoid')(merge)

    model = Model(inputs=[input_elmo, input_sentiment], outputs=output)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


# 세번째 모델
# ELMO 256차원 + CNN 100차원
def build_model_cnn(senti_shape):
    input_ELMo = layers.Input(shape=(1,), dtype="string")
    input_Sentiment = layers.Input(shape=(senti_shape, 1), dtype="float")

    embedding = ElmoEmbeddingLayer()(input_ELMo)
    dense_elmo = layers.Dense(256, activation='relu')(embedding)

    cnn_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(input_Sentiment)
    cnn_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(cnn_layer)
    cnn_layer = layers.Dropout(0.5)(cnn_layer)
    cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
    cnn_layer = layers.Flatten()(cnn_layer)
    cnn_layer = Dense(100, activation='relu')(cnn_layer)

    merge = concatenate([dense_elmo, cnn_layer])
    pred = layers.Dense(1, activation='sigmoid')(merge)

    model = Model(inputs=[input_ELMo, input_Sentiment], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def create_callbacks(model_dir):
    tensorboard_callback = TensorBoard(log_dir="./tensor", write_graph=True, write_grads=False)
    checkpoint_callback = ModelCheckpoint(filepath=model_dir + "/model-weights.{epoch:02d}-{val_acc:.6f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True)

    return [tensorboard_callback, checkpoint_callback]


def evaluate(prediction, test_y):
    y_pred = (prediction > 0.5)

    print(classification_report(test_y, y_pred, target_names=['0', '1']))
    print(accuracy_score(test_y, y_pred))

    index, count = np.unique(y_pred, return_counts=True)
    print(index)
    print(count)

    print(confusion_matrix(test_y, y_pred))


def main():
    train_test_dir = "../movie_persona/plot_data/critic_small_total.pickle"
    train_dir = "../movie_persona/plot_data/critic_small_train.pickle"
    test_dir = "../movie_persona/plot_data/critic_small_test.pickle"
    val_dir = test_dir

    model_dir = '../movie_persona/model/critic_test_lstm'

    # plot text 데이터 가공
    DataSet = load_data(train_test_dir, train_dir, test_dir, val_dir)

    train_x, val_x, test_x = DataSet['train']['plot'], DataSet['val']['plot'], DataSet['test']['plot']
    train_y, val_y, test_y = DataSet["train_y"], DataSet['val_y'], DataSet['test_y']
    train_x, val_x, test_x = train_x.tolist(), val_x.tolist(), test_x.tolist()

    train_x = [' '.join(t.split()[0:500]) for t in train_x]
    train_x = np.array(train_x, dtype=object)[:, np.newaxis]

    val_x = [' '.join(t.split()[0:500]) for t in val_x]
    val_x = np.array(val_x, dtype=object)[:, np.newaxis]

    test_x = [' '.join(t.split()[0:500]) for t in test_x]
    test_x = np.array(test_x, dtype=object)[:, np.newaxis]


    # sentiment score 데이터 가공
    senti_shape = 250
    train_sentiment, val_sentiment, test_sentiment = sentiment_processing(DataSet['train'], senti_shape), sentiment_processing(DataSet['val'], senti_shape), sentiment_processing(DataSet['test'], senti_shape)
    train_sentiment, val_sentiment, test_sentiment = train_sentiment['sentiment'], val_sentiment['sentiment'], test_sentiment['sentiment']

    train_sentiment, val_sentiment, test_sentiment = pd.DataFrame(train_sentiment), pd.DataFrame(val_sentiment), pd.DataFrame(test_sentiment)
    train_sentiment, val_sentiment, test_sentiment = np.array(train_sentiment), np.array(val_sentiment), np.array(test_sentiment)

    train_sentiment = train_sentiment.reshape(len(train_sentiment), len(train_sentiment[0]), 1)
    val_sentiment = val_sentiment.reshape(len(val_sentiment), len(val_sentiment[0]), 1)
    test_sentiment = test_sentiment.reshape(len(test_sentiment), len(test_sentiment[0]), 1)


    ### Model Parameter 설정
    batch_size = 32
    epoch = 30
    callbacks = create_callbacks(model_dir)


    # ### 모델1
    # model = build_model_elmo()
    # model.fit(x=train_x, y=train_y, epochs=epoch, batch_size=batch_size, validation_data=(test_x, test_y), callbacks=callbacks)


    ### 모델2: ELMo + BI_LSTM
    model = build_model_lstm(senti_shape)
    model.fit(x=[train_x, train_sentiment], y=train_y, epochs=epoch, batch_size=batch_size, validation_data=([test_x, test_sentiment], test_y), callbacks=callbacks)


    # ### 모델3 : ELMo + CNN
    # model = build_model_cnn(senti_shape)
    # model.fit(x=[train_x, train_sentiment], y=train_y, epochs=epoch, batch_size=batch_size, validation_data=([test_x, test_sentiment], test_y), callbacks=callbacks)


    ### F1 Score 구하기
    test_set = [test_x, test_sentiment]
    prediction = model.predict(test_set)
    evaluate(prediction, test_y)


if __name__ == '__main__':
    main()
