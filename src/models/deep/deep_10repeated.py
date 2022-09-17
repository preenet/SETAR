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
from keras.callbacks import EarlyStopping
from keras.models import load_model
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
elif dataset_name == 'kt':
    Xo, yo = joblib.load(Path.joinpath(root, configs['data']['kaggle_kt']))
elif dataset_name == 'tt':
    Xo, yo = joblib.load(Path.joinpath(root, configs['data']['kaggle_tt']))
elif dataset_name == 'to':
    data = joblib.load(Path.joinpath(root, configs['data']['kaggle_to']))
    Xo = data[0]
    yo = data[1]
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
for item in range(0, 2):
    file = open(configs['output_scratch'] + method + "_10repeated_" + str(dataset_name) + "_final1.csv" , "a")
    X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=item, stratify=yo)
    X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)
    num_class = np.unique(y).shape[0]
    
    recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=num_class)
    auc = tf.keras.metrics.AUC()
    f1 = tfa.metrics.F1Score(num_classes=num_class, average='macro')
    
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
    
 
    # getting training performance
    best_model = load_model(model_path + '/best_model_h5/cnn_to/' + method +'_' + str(dataset_name) + '_best_model_' +str(item)+'.h5', custom_objects={"F1Score": f1})

    acc, pre, rec, mcc, auc, f1 = test_deep(best_model, X_val_ps, yv, num_class)
    file.write(str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1))
    
    # for the final model, we train with (train+valid set) and test with test set

    with open(model_path + '/best_model_h5/cnn_to/' + method +'_' + str(dataset_name) + '_best_model_' +str(item)+'.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    auc = tf.keras.metrics.AUC()
    mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=np.unique(y).shape[0])
    f1 = tfa.metrics.F1Score(num_classes=np.unique(y).shape[0], average='macro')
    adam = tf.keras.optimizers.Adam(learning_rate=config['learn_rate']['value'])
    best_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy' , precision, recall, mcc, auc, f1])
    
    best_model.fit(np.vstack((X_train_ps, X_val_ps)), np.vstack((y_c, yv_c)),
                                    batch_size= config['batch_size']['value'],
                                    epochs= config['epochs']['value'],
                                    validation_data=(X_test_ps, yt_c)) 
                                   # callbacks=[EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)])
    best_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy' , precision, recall, mcc, auc, f1])
    
    # test with test set
    acc, pre, rec, mcc, auc, f1 = test_deep(best_model, X_test_ps, yt, num_class)
    file.write("," + str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1) + "\n")
file.close()
