from pathlib import Path

import numpy as np
import pandas as pd
import src.utilities as utils
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from src.models.metrics import test_deep
from tensorflow.keras.utils import to_categorical
from transformers import (BertConfig, BertTokenizer,
                          TFBertForSequenceClassification)

configs = utils.read_config()
root = utils.get_project_root()

df_ds = pd.read_csv(Path.joinpath(root, configs['data']['processed_to']))

y_ds = df_ds['target'].astype('category').cat.codes
yo = y_ds.to_numpy()
Xo = df_ds['processed']

tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True)
tokenizer.fit_on_texts(Xo)
sequences_train_num = tokenizer.texts_to_sequences(Xo)
#max_len = max([len(w) for w in sequences_train_num])
max_len = 64
sequences_train_num = tf.keras.preprocessing.sequence.pad_sequences(sequences_train_num, maxlen=max_len)

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def tokenize(sentences, tokenizer):
    input_ids, input_masks, input_segments = [],[],[]
    for sentence in sentences:
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, 
                                             return_attention_mask=True, return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])        
        
    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(Xo,yo,test_size = 0.2)

bert_train = tokenize(X_train, tokenizer)
bert_val = tokenize(X_val, tokenizer)


# Seed value
# Apparently you may use different seed values at each stage
seed_value= 1

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

# tf.random.set_seed(seed_value)
# for later versions: 
tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K

#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
# for later versions:
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

from transformers import BertConfig, TFBertModel


def create_model():
    config = BertConfig.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
    #transformer_model = AutoModel.from_pretrained('bert-base-multilingual-cased', config=config)
    transformer_model = TFBertModel.from_pretrained("bert-base-multilingual-cased", config=config)    
    input_ids = tf.keras.layers.Input(shape=(max_len,), name='input_token', dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(max_len,), name='masked_token', dtype=tf.int32)
    # No token_type layer

    embedding_layer = transformer_model(input_ids, attention_mask=attention_mask)[0]
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedding_layer)
    X = tf.keras.layers.GlobalMaxPool1D()(X)
    X = tf.keras.layers.Dense(8, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.4)(X)
    X = tf.keras.layers.Dense(1, activation='sigmoid')(X)
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs = X)

    for layer in model.layers[:3]:
      layer.trainable = False
    return model

model = create_model()    
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

def init(seed):
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.keras.backend.clear_session()
    
    init(0)   
ep = 80
bs = int(len(X_train))
history = model.fit(bert_train, y_train, validation_data=(bert_val, y_val), batch_size=bs, epochs=ep) 
