"""
1. run wandb <sweep name> sweep.yaml for hyperparam tuning at the src.model directory
2. run the link from wandb
3. get the best model from api or local or cloud, and run CNN_10repeated.py
4. run generate_report_10repeated_deepbaseline from src.utilities
5. copy results to the output folder
pree.t@cmu.ac.th
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import src.utilities as utils
import tensorflow as tf
import tensorflow_addons as tfa
from gensim.models import Word2Vec
from keras.callbacks import EarlyStopping
from keras.layers import (LSTM, Bidirectional, Dense, Dropout, Embedding,
                          Flatten, Input)
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
#from pythainlp import word_vector
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from wandb.keras import WandbCallback

import wandb

configs = utils.read_config()
root = utils.get_project_root()
model_path = str(Path.joinpath(root, configs['models']))

EMBEDDING_DIM= 300
MAX_SEQUENCE_LENGTH = 500
#########################################################################

dataset_name = 'kt'
if dataset_name == 'ws':
    Xo, yo = joblib.load(Path.joinpath(root, configs['data']['kaggle_ws']))
    w2v = Word2Vec.load(model_path+ '/' + 'w2v_ws_thwiki300.word2vec')
elif dataset_name == 'kt':
    Xo, yo = joblib.load(Path.joinpath(root, configs['data']['kaggle_kt']))
    w2v = Word2Vec.load(model_path+ '/' + 'w2v_kt_thwiki300_300.word2vec')
elif dataset_name == 'tt':
    Xo, yo = joblib.load(Path.joinpath(root, configs['data']['kaggle_tt']))
    w2v = Word2Vec.load(model_path+ '/' + 'w2v_tt_thwiki300.word2vec')
elif dataset_name == 'to':
    data = joblib.load(Path.joinpath(root, configs['data']['kaggle_to']))
    Xo = data[0]
    yo = data[1]
    w2v = Word2Vec.load(model_path+ '/' + 'to_thwiki300.word2vec')
else: 
    print("No such dataset.")
    sys.exit(-1)
seed = 0
#########################################################################

# print("Building w2v model...")

# w2v = Word2Vec(vector_size=300, min_count=1, window = 5, workers=8)
# w2v.build_vocab(df_ds['processed'])
# w2v.train(df_ds['processed'], total_examples=w2v.corpus_count, epochs=100)

# w2v_thwiki = word_vector.get_model()
# w2v.build_vocab(w2v_thwiki.index_to_key, update=True)
# w2v.wv.vectors_lockf = np.ones(len(w2v.wv))
# w2v.wv.intersect_word2vec_format(model_path+ '/' + 'thai2vec.bin', binary=True, lockf=1.0)


# get weight from word2vec as a keras embedding metric
keyed_vectors = w2v.wv  
weights = keyed_vectors.vectors  
w2v_keras_layer = Embedding(
    input_dim=weights.shape[0],
    output_dim=weights.shape[1],
    weights=[weights],
    trainable=True # (false for fixed)
)

wandb.init(settings=wandb.Settings(_disable_stats=True))
config = wandb.config


X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=seed, stratify=yo)
X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)

num_class = np.unique(y).shape[0]
recall = tf.keras.metrics.Recall()
precision = tf.keras.metrics.Precision()
auc = tf.keras.metrics.AUC()
mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=np.unique(y).shape[0])
f1 = tfa.metrics.F1Score(num_classes=np.unique(y).shape[0], average='macro')
adam = tf.keras.optimizers.Adam(lr=config.learn_rate)

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

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=12)

model = Sequential()
model.add(Input(shape=(MAX_SEQUENCE_LENGTH,)))
model.add(w2v_keras_layer)
model.add(Bidirectional(LSTM(config.layer_1_size, return_sequences = True)))
model.add(Dropout(config.dropout))
model.add(Flatten()) 
model.add(Dense(config.hidden_layer_size, activation='relu')) 
model.add(Dense(num_class, activation='softmax'))

model.compile(
    loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy' , precision, recall, mcc, auc, f1])

model.fit(X_train_ps, y_c,  validation_data=(X_val_ps, yv_c),
          epochs=config.epochs,
          batch_size=config.batch_size,
          initial_epoch=wandb.run.step,  # for resumed runs
           callbacks=[WandbCallback(save_model=True, monitor="loss"), es])
