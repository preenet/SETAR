"""
10 repeated hold-out from the best models of 10 seeds (this script only for CNN and Bi-LSTM)
for BERT, use BERT_manaul.py
for WangchanBERTa, use wangchan_predict.py
"""

import os
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import src.utilities as utils
import tensorflow as tf
import tensorflow_addons as tfa
import yaml
from gensim.models import Word2Vec
from keras.callbacks import EarlyStopping
from keras.layers import (Conv1D, Dense, Dropout, Embedding, Flatten, Input,
                          MaxPooling1D)
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from src.models.metrics import test_deep
from tensorflow.keras.utils import to_categorical

configs = utils.read_config()
root = utils.get_project_root()
model_path = str(Path.joinpath(root, configs['models']))

EMBEDDING_DIM= 300
MAX_SEQUENCE_LENGTH = 500

#########################################################################
dataset_name = 'to'
method = 'cnn'


if dataset_name == 'ws':
    Xo, yo = joblib.load(Path.joinpath(root, configs['data']['kaggle_ws']))
    w2v = Word2Vec.load(model_path+ '/' + 'w2v_ws_thwiki300.word2vec')
elif dataset_name == 'kt':
    Xo, yo = joblib.load(Path.joinpath(root, configs['data']['kaggle_kt']))
    w2v = Word2Vec.load(model_path+ '/' + 'w2v_kt_thwiki300_300.word2vec')
elif dataset_name == 'tt':
    Xo, yo = joblib.load(Path.joinpath(root, configs['data']['kaggle_tt']))
    w2v = Word2Vec.load(model_path+ '/' + 'w2v_tt_thwiki300_300.word2vec')
elif dataset_name == 'to':
    data = joblib.load(Path.joinpath(root, configs['data']['kaggle_to']))
    Xo = data[0]
    yo = data[1]
    w2v = Word2Vec.load(model_path+ '/' + 'to_thwiki300.word2vec')
else: 
    print("No such dataset.")
    sys.exit(-1)
#########################################################################

def init(seed):
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set as {seed}")
    
init(0)

keyed_vectors = w2v.wv  
weights = keyed_vectors.vectors  
w2v_keras_layer = Embedding(
    input_dim=weights.shape[0],
    output_dim=weights.shape[1],
    weights=[weights],
    trainable=True
)


for item in range(0, 1):
    file = open(configs['output_scratch'] + method + "_10repeated_" + str(dataset_name) + "_final2.csv" , "a")
    X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=item, stratify=yo)
    X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)
    num_class = np.unique(y).shape[0]

    recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    auc = tf.keras.metrics.AUC()
    mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=np.unique(y).shape[0])
    f1 = tfa.metrics.F1Score(num_classes=np.unique(y).shape[0], average='macro')

    
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

    best_model = load_model(model_path + '/best_model_deep/cnn_to/' + method +'_' + str(dataset_name) + '_best_model_' +str(item)+'.h5')

    acc, pre, rec, mcc, auc, f1 = test_deep(best_model, X_val_ps, yv, num_class)
    file.write(str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1))
    
    acc, pre, rec, mcc, auc, f1 = test_deep(best_model, X_test_ps, yt, num_class)
    file.write("," + str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1) + "\n")

file.close()
