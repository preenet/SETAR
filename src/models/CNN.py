import sys
import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers

import src.utilities as utils
import src.feature.build_features as bf

config = utils.read_config()
df_ds = pd.read_csv(config['data']['processed_ws'])
y_ds = df_ds['target'].astype('category').cat.codes

Xo = df_ds['processed']
yo = y_ds.to_numpy()
dict = bf.get_dict_vocab()

iname = sys.argv[1]

for item in range(0, 10):
    X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=item, stratify=yo)
    X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)
    
    
    fe, X_train_val = bf.extract(iname, X_train, (1,1))
    X_train_val = X_train_val
    X_val_val = fe.transform(X_val)
    X_test_val = fe.transform(X_test)
    
    
    EMBEDDING_DIM=300
    maxlen = 300

    model = Sequential()
    model.add(layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=maxlen))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    history = model.fit(X_train, yv,
                        epochs=10,
                        validation_data=(X_test, yt),
                        batch_size=10)