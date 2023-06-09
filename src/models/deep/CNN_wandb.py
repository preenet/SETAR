"""
This script response to find the optimal hyper-parameters of CNN 
1. run wandb sweep [sweep.yaml] for hyperparam tuning at the src/models directory
2. run the link that generated from wandb
3. get the best model from api or local and run the script deep_10repeated.py for each fold.
4. run the script to generate_report_10repeated_deepbaseline from src.utilities
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
import src.utilities as utils
import tensorflow as tf
import tensorflow_addons as tfa
from gensim.models import Word2Vec
from keras.callbacks import EarlyStopping
from keras.layers import (Conv1D, Dense, Dropout, Embedding, Flatten, Input,
                          MaxPooling1D, concatenate, merge)
from keras.models import Model, Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from pythainlp import word_vector
from sklearn.model_selection import train_test_split
from src.models.metrics import test_deep
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



#########################################################################

dataset_name = 'to'
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
#item = 3
method = 'cnn'
#########################################################################
# tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True)
# tokenizer.fit_on_texts(Xo)
# sequences_train_num = tokenizer.texts_to_sequences(Xo)
# MAX_SEQUENCE_LENGTH = max([len(w) for w in sequences_train_num])
MAX_SEQUENCE_LENGTH =500
# print("Building w2v model...")
# tok_train = [text.split() for text in Xo]
# w2v = Word2Vec(vector_size=300, min_count=1, window = 5, workers=8)
# w2v.build_vocab(tok_train )
# w2v_thwiki = word_vector.get_model()
# w2v.build_vocab(w2v_thwiki.index_to_key, update=True)
# w2v.wv.vectors_lockf = np.ones(len(w2v.wv))
# w2v.wv.intersect_word2vec_format(model_path+ '/' + 'thai2vec.bin', binary=True, lockf=1.0)
# w2v.train(tok_train, total_examples=w2v.corpus_count, epochs=100)

# Word2Vec.save(w2v, model_path+ '/' + 't0_thwiki300.word2vec')

# make sure to load a proper word2vec model according to the dataset.

#get weight from word2vec as a keras embedding metric
keyed_vectors = w2v.wv  
weights = keyed_vectors.vectors  
w2v_keras_layer = Embedding(
    input_dim=weights.shape[0],
    output_dim=weights.shape[1],
    weights=[weights],
    trainable=True
)

train_acc = []
test_acc = []

config_d = dict(
    dropout=0.5,
    batch_size=32,
    epochs = 20,
    learn_rate=0.005,
    )

for item in range(0, 10):
    init(0)
    # wandb.init(settings=wandb.Settings(_disable_stats=True))
    # config = wandb.config
    X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=item, stratify=yo)
    X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)
    num_class = np.unique(y).shape[0]

    recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    auc = tf.keras.metrics.AUC()
    mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=np.unique(y).shape[0])
    f1 = tfa.metrics.F1Score(num_classes=np.unique(y).shape[0], average='macro')
    adam = tf.keras.optimizers.Adam(learning_rate= config_d['learn_rate'])

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
    yt_c = to_categorical(yt)
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)

    # construct basic CNN arch based on the paper Pasupal et al.
    # model = Sequential()
    # model.add(Input(shape=(MAX_SEQUENCE_LENGTH,)))
    # model.add(w2v_keras_layer)
    # model.add(Conv1D(config.layer_1_size, kernel_size=3, padding='same', activation='relu'))
    # model.add(MaxPooling1D())
    # model.add(Dropout(config.dropout))
    # model.add(Flatten()) 
    # model.add(Dense(config.hidden_layer_size, activation='relu'))
    # model.add(Dense(num_class, activation='softmax'))

    # construct parallel pooling layers
    num_filters=3
    filter_sizes = [3,4,5]

    model = Model()
    emb = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = (w2v_keras_layer)(emb)

    pooled_outputs = []
    for i in range(len(filter_sizes)):
        conv = Conv1D(num_filters, kernel_size=filter_sizes[i], padding='valid', activation='relu')(x)
        conv = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH-filter_sizes[i]+1, strides=1, padding = 'valid')(conv)         
        pooled_outputs.append(conv)
    merge = concatenate(pooled_outputs)
    x = Flatten()(merge)
    x = Dropout(config_d['dropout'])(x)

    out = Dense(num_class, activation= 'softmax')(x)
    model = Model(inputs=emb,outputs=out)
    print(model.summary())

    model.compile(
        loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy' , precision, recall, mcc, auc, f1])
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=3)
    #callback= WandbCallback(save_model=True, monitor='val_loss', mode='min')

    model.fit(X_train_ps, y_c,  validation_data=(X_val_ps, yv_c),
            epochs=config_d['epochs'],
            batch_size=config_d['batch_size'], callbacks=[es])
           #callbacks=[WandbCallback(save_model=True, monitor="loss"), es])
           

    file = open(method + "_10repeated_" + str(dataset_name) + "_final3.csv" , "a")
    acc, pre, rec, mcc, auc, f1 = test_deep(model, X_val_ps, yv, num_class)
    train_acc.append(acc)
    file.write(str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1))

    acc, pre, rec, mcc, auc, f1 = test_deep(model, X_test_ps, yt, num_class)
    test_acc.append(acc)
    file.write("," + str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1) + "\n")
    
    print("train_acc: ", np.mean(train_acc), " test_acc: ", np.mean(test_acc))
    model.save( method +'_' + str(dataset_name) + '_best_model_' +str(item)+'.h5' )
    from keras import backend as K
    K.clear_session()
    del model
file.close()
