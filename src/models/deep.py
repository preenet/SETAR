
'''
We implemented our CNN and bi-LSTM based on the model arch and param search strategy of the paper: (Pasupa & Seneewong Na Ayutthaya, 2022) 
'''
import sys

import numpy as np
import pandas as pd
import src.utilities as utils
import talos
import tensorflow as tf
import tensorflow_addons as tfa
from gensim.models import Word2Vec
from keras.layers import (LSTM, Conv1D, Dense, Dropout, Embedding, Flatten,
                          Input, MaxPooling1D)
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

config = utils.read_config()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

EMBEDDING_DIM= 300
MAX_SEQUENCE_LENGTH = 500

df_ds = pd.read_csv(config['data']['processed_tt'])

y_ds = df_ds['target'].astype('category').cat.codes
yo = y_ds.to_numpy()
Xo = df_ds['processed']

arch = sys.argv[1]
data_name = sys.argv[2]
print(arch, ", ", data_name)
w2v = Word2Vec.load(config['models'] + 'w2v_ws_thwiki300.word2vec')

keyed_vectors = w2v.wv  
weights = keyed_vectors.vectors  
w2v_keras_layer = Embedding(
    input_dim=weights.shape[0],
    output_dim=weights.shape[1],
    weights=[weights],
    trainable=True
)

X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=0, stratify=yo)
X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0, stratify=y_tmp)

num_class = np.unique(y).shape[0]

tokenizer  = Tokenizer(num_words = MAX_SEQUENCE_LENGTH)
tokenizer.fit_on_texts(X_train)
train_sequences =  tokenizer.texts_to_sequences(X_train)
valid_sequences = tokenizer.texts_to_sequences(X_val)
test_sequences = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index)
print("vocab size is:", vocab_size)
word_index = tokenizer.word_index


# pad dataset to a maximum review length in words
X_train_ps = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_val_ps = pad_sequences(valid_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_test_ps = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(X_train_ps.shape, X_val_ps.shape, X_test_ps.shape)

y_c = to_categorical(y)
yv_c = to_categorical(yv)
yt_c = to_categorical(yt)

para = {'dropout':[0.1, 0.2, 0.3, 0.4, 0.5],
        'num_neurons': [16, 32, 64, 256, 512],
        'batch_size': [16, 32, 64],
        'epochs': [30]
        }

def build_model(X_train_ps, y_c, X_val_ps, yv_c, params):
    model = Sequential()
    model.add(Input(shape=(MAX_SEQUENCE_LENGTH,)))
    model.add(w2v_keras_layer)
    model.add(Conv1D(params['num_neurons'], 3, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(num_class, activation='softmax'))
    
    f1 = tfa.metrics.F1Score(num_classes=np.unique(y).shape[0], average='macro')
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy' , f1])


    out = model.fit(X_train_ps, y_c,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        validation_data=(X_val_ps, yv_c),
                        verbose=0,
                        callbacks=[talos.utils.early_stopper(params['epochs'], monitor='val_accuracy', mode='moderate')])

    return out, model


history = talos.Scan(x=X_train_ps, y=y_c, model=build_model, params=para, experiment_name=arch)

r = talos.Reporting(history)
print("Best model: ", r.high('val_accuracy'))
print('Experiment terminated properly!')
