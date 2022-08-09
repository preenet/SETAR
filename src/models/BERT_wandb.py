from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import src.utilities as utils
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from src.feature.process_thai_text import process_text
from src.models.metrics import test_deep
from tensorflow.keras.utils import to_categorical
from transformers import (BertConfig, BertTokenizer,
                          TFBertForSequenceClassification)

configs = utils.read_config()
root = utils.get_project_root()
from wandb.keras import WandbCallback

import wandb

wandb.init(settings=wandb.Settings(_disable_stats=True))
config = wandb.config

df_ds = pd.read_csv(Path.joinpath(root, configs['data']['processed_ws']))
#df_ds = df_ds[df_ds['processed'].str.len() < 320]
y_ds = df_ds['target'].astype('category').cat.codes
yo = y_ds.to_numpy()
#Xo = df_ds['processed']
Xo = [' '.join(process_text(item))  for item in df_ds['text'].apply(str)]


tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True)
tokenizer.fit_on_texts(Xo)
sequences_train_num = tokenizer.texts_to_sequences(Xo)
#max_len = max([len(w) for w in sequences_train_num])
max_len = 128
print("Max length is:", max_len)
sequences_train_num = tf.keras.preprocessing.sequence.pad_sequences(sequences_train_num, maxlen=max_len )

bert = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(bert)

def tokenize(sentences, tokenizer):
    input_ids, input_masks = [],[]
    for sentence in sentences:
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, 
                                             return_attention_mask=True, return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])       
        
    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')



# file = open(configs['output_scratch'] +"bert_to.csv", "a")
X_train, X_tmp, y_train, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=0, stratify=yo)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0, stratify=y_tmp)

num_class = np.unique(yo).shape[0]
y_train_c = to_categorical(y_train)
y_val_c = to_categorical(y_val)
#y_test_c = to_categorical(yt)



bert_train = tokenize(X_train, tokenizer)
bert_val = tokenize(X_val, tokenizer)
#bert_test = tokenize(X_test, tokenizer)

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

    embedding_layer = transformer_model(input_ids, attention_mask=attention_mask)
    X = embedding_layer[1] # for classification we only care about the pooler output
    
    X = tf.keras.layers.Dense(32, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.2)(X)
    X = tf.keras.layers.Dense(4, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs = X)

    # for layer in model.layers[:3]:
    #   layer.trainable = False
    
    return model

# def create_model_finetune():
#     # Fine-tuning a Pretrained transformer model
#     loss = tf.keras.losses.CategoricalCrossentropy()
#     accuracy = tf.keras.metrics.CategoricalAccuracy()
#     recall = tf.keras.metrics.Recall()
#     precision = tf.keras.metrics.Precision()
#     auc = tf.keras.metrics.AUC()
#     mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=np.unique(y).shape[0])
#     f1 = tfa.metrics.F1Score(num_classes=num_class, average='macro')
    
#     configuration = BertConfig.from_pretrained(bert)

#     configuration.output_hidden_states = True
#     transformer_model = TFBertForSequenceClassification.from_pretrained(bert, config = configuration)

#     input_ids_layer = tf.keras.layers.Input(shape=(max_len, ), dtype=tf.int32)
#     input_mask_layer = tf.keras.layers.Input(shape=(max_len ), dtype=tf.int32)

#     embeddings = transformer_model([input_ids_layer, input_mask_layer])[0]
#     dense_output = tf.keras.layers.Dense(num_class, activation='softmax')(embeddings ) 
#     model = tf.keras.Model(inputs=[input_ids_layer, input_mask_layer], outputs=dense_output)
    
#     # for layer in model.layers[:2]:
#     #     layer.trainable = False
    
#     model.compile(tf.keras.optimizers.Adam(lr=3e-5), loss=loss, metrics=[accuracy, precision, recall, mcc, auc, f1])
#     return model

# model = create_model_finetune()
# model.summary()


def init(seed):
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.keras.backend.clear_session()
    
model = create_model()    
recall = tf.keras.metrics.Recall()
precision = tf.keras.metrics.Precision()
f1 = tfa.metrics.F1Score(num_classes=4, average='macro')
mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=4)
auc = tf.keras.metrics.AUC()
adam = tf.keras.optimizers.Adam(learning_rate=config.learn_rate)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy' , precision, recall, mcc, auc, f1])
#model.compile(tf.keras.optimizers.Adam(lr=3e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

init(0)  
ep = config.epochs
bs = 32
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
hist = model.fit(bert_train, y_train_c, validation_data=(bert_val, y_val_c), batch_size=bs, epochs=ep, verbose=1,callbacks=[WandbCallback(save_model=False, monitor="loss"), es]) 

# file.write( str(0) + "," + str(max(hist.history['val_accuracy'])) + "," + str(max(hist.history['val_precision'])) + \
#         "," + str(max(hist.history['val_recall'])) + ","  + str(max(hist.history['val_MatthewsCorrelationCoefficient'])) + \
#             ","  + str(max(hist.history['val_auc'])) + "," + str(max(hist.history['val_f1_score'])))
    
# # test with test set
# acc, pre, rec, mcc, auc, f1 = test_deep(model, bert_val , y_val)
# file.write("," + str(0) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1) + "\n")

# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set()
# plt.figure(num=None, figsize=(16, 8), dpi=90, facecolor='w', edgecolor='k')
# plt.plot()
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('model accuracy')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()

