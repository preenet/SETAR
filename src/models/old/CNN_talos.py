
'''
We implemented our CNN based on the model arch and param search strategy of the paper: (Pasupa & Seneewong Na Ayutthaya, 2022) 
'''

import sys

import numpy as np
import pandas as pd
import src.utilities as utils
import talos
import tensorflow as tf
from gensim.models import Word2Vec
from keras.layers import (Conv1D, Dense, Dropout, Embedding, Flatten, Input,
                          MaxPooling1D)
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot
from pythainlp import word_vector
from sklearn.model_selection import train_test_split
from src.models.LRFind import LRFind
from src.models.metrics import test_deep
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
#from pythainlp import word_vector

# create word2vec for kt corpus
print("Building w2v model...")
w2v = Word2Vec(vector_size=300, min_count=1, window = 5, workers=8)
w2v.build_vocab(df_ds['processed'])
w2v.train(df_ds['processed'], total_examples=w2v.corpus_count, epochs=100)

w2v_thwiki = word_vector.get_model()
w2v.build_vocab(w2v_thwiki.index_to_key, update=True)
w2v.wv.vectors_lockf = np.ones(len(w2v.wv))
w2v.wv.intersect_word2vec_format(config['models']+'thai2vec.bin', binary=True, lockf=1.0)


# w2v = Word2Vec.load(config['models'] + 'w2v_tt_thwiki300.word2vec')

# get weight from word2vec as a keras embedding metric
keyed_vectors = w2v.wv  
weights = keyed_vectors.vectors  
w2v_keras_layer = Embedding(
    input_dim=weights.shape[0],
    output_dim=weights.shape[1],
    weights=[weights],
    trainable=True
)


# we build one channel cnn here.
def build_model(X_train_ps, y_c, X_val_ps, yv_c, params):
        model = Sequential()
        model.add(Input(shape=(MAX_SEQUENCE_LENGTH,)))
        model.add(w2v_keras_layer)
        model.add(Conv1D(filters=params['num_neurons'], kernel_size=(3,), padding='same', activation='relu'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dropout(params['dropout']))
        model.add(Dense(num_class, activation='softmax'))
        
        optimizer = tf.keras.optimizers.Adam(lr=params['lr'])
        model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy' , talos.utils.metrics.precision,
                            talos.utils.metrics.recall,
                            talos.utils.metrics.matthews,
                            talos.utils.metrics.f1score])

        
        out = model.fit(X_train_ps, y_c,
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            validation_data=(X_val_ps, yv_c),
                            verbose=0,
                            callbacks=[talos.utils.early_stopper(params['epochs'], monitor='val_accuracy', mode='moderate')])

        return out, model
    

for item in range(0, 10):
    file = open(config['output_scratch'] +arch+ "_" + data_name + ".csv", "a")

    X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=item, stratify=yo)
    X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)

    num_class = np.unique(y).shape[0]

    tokenizer  = Tokenizer(num_words = MAX_SEQUENCE_LENGTH)
    tokenizer.fit_on_texts(X_train)
    train_sequences =  tokenizer.texts_to_sequences(X_train)
    valid_sequences = tokenizer.texts_to_sequences(X_val)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    vocab_size = len(tokenizer.word_index)
    #print("vocab size is:", vocab_size)
    word_index = tokenizer.word_index


    # pad dataset to a maximum review length in words
    X_train_ps = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    X_val_ps = pad_sequences(valid_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    X_test_ps = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print(X_train_ps.shape, X_val_ps.shape, X_test_ps.shape)

    y_c = to_categorical(y)
    yv_c = to_categorical(yv)
    yt_c = to_categorical(yt)

    para = {'dropout':[0.4, 0.5],
        'num_neurons': [16, 32, 64, 128, 256, 512],
        'batch_size': [16, 32, 64],
        'epochs': [32, 64],
        'lr':[0.01,0.001,0.0001]
        }

    history = talos.Scan(x=X_train_ps, y=y_c, x_val=X_val_ps, y_val=yv_c, model=build_model, params=para, experiment_name=arch)

    r = talos.Reporting(history)
    print("Best model: ", r.high('val_accuracy'))

    a = talos.Analyze(history)
    a_table = a.table('val_loss', sort_by='val_accuracy', exclude=['start', 'end', 'duration'])
    #print('a_table = \n{}'.format(a_table))

    best_model = history.best_model(metric='val_accuracy', asc=False)

    acc, pre, rec, mcc, auc, f1 = test_deep(best_model, X_val_ps, yv)
    file.write(str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1))

    # fit model again with best params and use training and validation set for training
    model = Sequential()
    model.add(Input(shape=(MAX_SEQUENCE_LENGTH,)))
    model.add(w2v_keras_layer)
    model.add(Conv1D(filters=int(a_table.iloc[0]['num_neurons']), kernel_size=(3,), padding='same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dropout(a_table.iloc[0]['dropout']))
    model.add(Dense(num_class, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(lr=a_table.iloc[0]['lr'])
    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy' , talos.utils.metrics.precision,
                        talos.utils.metrics.recall,
                        talos.utils.metrics.matthews,
                        talos.utils.metrics.f1score])

    model.fit(np.vstack((X_train_ps, X_val_ps)), np.vstack((y_c, yv_c)),
                                batch_size= int(a_table.iloc[0]['batch_size']),
                                epochs= int(a_table.iloc[0]['epochs']),
                                validation_data=(X_test_ps, yt_c),
                                verbose=0, 
                                callbacks=[talos.utils.early_stopper(int(a_table.iloc[0]['epochs']), monitor='val_accuracy', mode='moderate')])
    acc, pre, rec, mcc, auc, f1 = test_deep(model, X_test_ps, yt)
    file.write("," + str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1) + "\n")
file.close()

print('Experiment terminated properly!')
