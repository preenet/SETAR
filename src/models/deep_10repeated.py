"""
run repeated 10 fold from the best model
"""

from pathlib import Path

import numpy as np
import pandas as pd
import src.utilities as utils
import tensorflow as tf
import tensorflow_addons as tfa
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

df_ds = pd.read_csv(Path.joinpath(root, configs['data']['processed_ws']))

y_ds = df_ds['target'].astype('category').cat.codes
yo = y_ds.to_numpy()
Xo = df_ds['processed']


for item in range(0, 10):
    file = open(configs['output_scratch'] +"blstm_10repeated_kt.csv", "a")
    X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=item, stratify=yo)
    X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)
    num_class = np.unique(y).shape[0]
    
    recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    auc = tf.keras.metrics.AUC()
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

    # load best model
    best_model = load_model(model_path + '/best_model_h5/' + 'blstm_ws_best_model.h5', custom_objects={"F1Score": f1})

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
    # train with (train+valid set)
    hist = best_model.fit(np.vstack((X_train_ps, X_val_ps)), np.vstack((y_c, yv_c)),
                                    batch_size= 64,
                                    epochs= 60,
                                    validation_data=(X_test_ps, yt_c),
                                    verbose=1,
                                    callbacks=[es])
    file.write( str(item) + "," + str(max(hist.history['val_accuracy'])) + "," + str(max(hist.history['val_precision'])) + \
        "," + str(max(hist.history['val_recall'])) + ","  + str(max(hist.history['val_MatthewsCorrelationCoefficient'])) + \
            ","  + str(max(hist.history['val_auc'])) + "," + str(max(hist.history['val_f1_score'])))
    
    # test with test set
    acc, pre, rec, mcc, auc, f1 = test_deep(best_model, X_test_ps, yt)
    file.write("," + str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1) + "\n")
file.close()
