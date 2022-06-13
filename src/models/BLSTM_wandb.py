"""
1. run wandb <sweep name> sweep.yaml for hyperparam tuning at the src.model directory
2. run the link from wandb
3. get the best model from api or local and run CNN_10repeated.py
4. run generate_report_10repeated_deepbaseline from src.utilities
5. copy results to the output folder
pree.t@cmu.ac.th
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import src.utilities as utils
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from gensim.models import Word2Vec
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding, Input
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from pythainlp import word_vector
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from wandb.keras import WandbCallback

configs = utils.read_config()
root = utils.get_project_root()
model_path = str(Path.joinpath(root, configs['models']))

EMBEDDING_DIM= 300
MAX_SEQUENCE_LENGTH = 500

df_ds = pd.read_csv(Path.joinpath(root, configs['data']['processed_ws']))

y_ds = df_ds['target'].astype('category').cat.codes
yo = y_ds.to_numpy()
Xo = df_ds['processed']

print("Building w2v model...")

w2v = Word2Vec(vector_size=300, min_count=1, window = 5, workers=8)
w2v.build_vocab(df_ds['processed'])
w2v.train(df_ds['processed'], total_examples=w2v.corpus_count, epochs=100)

w2v_thwiki = word_vector.get_model()
w2v.build_vocab(w2v_thwiki.index_to_key, update=True)
w2v.wv.vectors_lockf = np.ones(len(w2v.wv))
w2v.wv.intersect_word2vec_format(model_path+ '/' + 'thai2vec.bin', binary=True, lockf=1.0)


w2v = Word2Vec.save(model_path+ '/' + 'w2v_ws_thwiki300_300.word2vec')

# get weight from word2vec as a keras embedding metric
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
    layer_1_size=16,
    learn_rate=0.001,
    epochs=64,
    )

resume = sys.argv[-1] == "--resume"
wandb.init(project="cnn-ws", config=defaults, resume=resume, settings=wandb.Settings(_disable_stats=True))
config = wandb.config

X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=0, stratify=yo)
X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0, stratify=y_tmp)
num_class = np.unique(y).shape[0]
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

# build model
if wandb.run.resumed:
    print("RESUMING")
    # restore the best model
    model = load_model(wandb.restore("cnn_ws_model-best.h5").name)
else:
    model = Sequential()
    model.add(Input(shape=(MAX_SEQUENCE_LENGTH,)))
    model.add(w2v_keras_layer)
    model.add(Bidirectional(config.layer_1_size, return_sequences = True))
    model.add(Dropout(config.dropout))
    model.add(Dense(config.hidden_layer_size, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(X_train_ps, y_c,  validation_data=(X_val_ps, yv_c),
          epochs=config.epochs,
          initial_epoch=wandb.run.step,  # for resumed runs
          callbacks=[WandbCallback()])

model.save(model_path+ '/' +"cnn_ws.h5")
