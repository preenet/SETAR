"""
This script response to find the optimal hyper-parameters of CNN 
1. run wandb sweep [sweep.yaml] for hyperparam tuning at the src/models directory
2. run the link that generated from wandb
3. get the best model from api or local and run the script deep_10repeated.py
4. run generate_report_10repeated_deepbaseline from src.utilities
5. copy results to the output folder.

# best model of each seed stored in the models folder at root folder.
pree.t@cmu.ac.th
"""

import os
import random
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
from keras.layers import (Conv1D, Dense, Dropout, Embedding, Flatten, Input,
                          MaxPooling1D)
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from pythainlp import word_vector
from sklearn.model_selection import train_test_split
from src.feature.process_thai_text import process_text
from tensorflow.keras.utils import to_categorical
from wandb.keras import WandbCallback

import wandb


def init(seed):
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set as {seed}")

configs = utils.read_config()
root = utils.get_project_root()
model_path = str(Path.joinpath(root, configs['models']))

MAX_SEQUENCE_LENGTH = 500

Xo, yo = joblib.load(Path.joinpath(root, configs['data']['kaggle_ws']))


# print("Building w2v model...")
# tok_train = [text.split() for text in Xo]
# w2v = Word2Vec(vector_size=300, min_count=1, window = 5, workers=8)
# w2v.build_vocab(tok_train )
# w2v_thwiki = word_vector.get_model()
# w2v.build_vocab(w2v_thwiki.index_to_key, update=True)
# w2v.wv.vectors_lockf = np.ones(len(w2v.wv))
# w2v.wv.intersect_word2vec_format(model_path+ '/' + 'thai2vec.bin', binary=True, lockf=1.0)
# w2v.train(tok_train, total_examples=w2v.corpus_count, epochs=100)

# Word2Vec.save(w2v, model_path+ '/' + 'tt_thwiki300.word2vec')

# make sure to load a proper word2vec model according to the dataset.
w2v = Word2Vec.load(model_path+ '/' + 'w2v_ws_thwiki300_300.word2vec')

#get weight from word2vec as a keras embedding metric
keyed_vectors = w2v.wv  
weights = keyed_vectors.vectors  
w2v_keras_layer = Embedding(
    input_dim=weights.shape[0],
    output_dim=weights.shape[1],
    weights=[weights],
    trainable=True
)

defaults = dict(
    dropout=0.5,
    hidden_layer_size=128,
    layer_1_size=64,
    learn_rate=0.001,
    batch_size = 64,
    epochs=80,
    )
wandb.init(settings=wandb.Settings(_disable_stats=True))
config = wandb.config

init(0)
X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=1, stratify=yo)
X_val, X_test, yv, _ = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=1, stratify=y_tmp)
num_class = np.unique(y).shape[0]

recall = tf.keras.metrics.Recall()
precision = tf.keras.metrics.Precision()
auc = tf.keras.metrics.AUC()
mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=np.unique(y).shape[0])
f1 = tfa.metrics.F1Score(num_classes=np.unique(y).shape[0], average='macro')
adam = tf.keras.optimizers.Adam(learning_rate=config.learn_rate)

tokenizer  = Tokenizer(num_words = MAX_SEQUENCE_LENGTH)
tokenizer.fit_on_texts(X_train)
train_sequences =  tokenizer.texts_to_sequences(X_train)
valid_sequences = tokenizer.texts_to_sequences(X_val)
test_sequences = tokenizer.texts_to_sequences(X_test)

# pad dataset to a maximum review length in words
X_train_ps = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_val_ps = pad_sequences(valid_sequences, maxlen=MAX_SEQUENCE_LENGTH)
X_test_ps = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(X_train_ps.shape, X_val_ps.shape, X_test_ps.shape)

y_c = to_categorical(y)
yv_c = to_categorical(yv)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)


model = Sequential()
model.add(Input(shape=(MAX_SEQUENCE_LENGTH,)))
model.add(w2v_keras_layer)
model.add(Conv1D(config.layer_1_size, kernel_size=3, activation='relu'))
model.add(MaxPooling1D())
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

