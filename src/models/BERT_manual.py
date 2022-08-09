
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import src.utilities as utils
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from src.feature.process_thai_text import process_text, process_text_old
from src.models.metrics import test_bert
from tensorflow.keras.utils import to_categorical
from transformers import (AutoConfig, AutoModel, AutoTokenizer, BertConfig,
                          BertTokenizer, TFAutoModel,
                          TFBertForSequenceClassification, TFBertModel)
from wandb.keras import WandbCallback

configs = utils.read_config()
root = utils.get_project_root()

#bert = 'bert-base-multilingual-cased'
bert = 'bert-base-th-cased'
#bert = 'bert-base-thai' 

df_ds = pd.read_csv(Path.joinpath(root, configs['data']['processed_to']))
df_ds = df_ds[df_ds['processed'].str.len() < 320]
y_ds = df_ds['target'].astype('category').cat.codes
yo = y_ds.to_numpy()


#joblib.dump((Xo, yo), "kt-bert-new-token.sav")   


# print("Max length is:", max_len)

# a custom tokenizer function and call the encode_plus method of the selected BERT tokenizer.
def tokenize(sentences, tokenizer):
    input_ids, input_masks = [],[]
    for sentence in sentences:
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, 
                                             return_attention_mask=True, return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])       
        
    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 1

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

# tf.random.set_seed(seed_value)
# for later versions: 
tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
# for later versions:
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

    
def init(seed):
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.keras.backend.clear_session()

def create_model(bert_model):
    
    input_ids = tf.keras.layers.Input(shape=(max_len,), name='input_token', dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(max_len,), name='masked_token', dtype=tf.int32)

    embedding_layer = bert_model(input_ids, attention_mask=attention_mask)
    X = embedding_layer[1] # for classification we only care about the pooler output
    
    X = tf.keras.layers.Dense(32, activation='relu')(X)
    #X = tf.keras.layers.Dropout(0.1)(X)
    X = tf.keras.layers.Dense(2, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs = X)

    # for layer in model.layers[:3]:
    #   layer.trainable = False
    return model

# Compute 10 repeated
#FIXME: model keep using the old weight for next loop
for item in range(0, 10):
    tf.keras.backend.clear_session()
   
    
    if bert == 'bert-base-thai':
        Xo = [process_text_old(item) for item in df_ds['text'].apply(str)]
        tokenizer = AutoTokenizer.from_pretrained(bert)
        bert_config = AutoConfig.from_pretrained("bert-base-thai", output_hidden_states=True)
        bert_model = AutoModel.from_pretrained('bert-base-thai',  bert_config)
    
    elif bert == 'bert-base-th-cased':
        Xo = [' '.join(process_text(item))  for item in df_ds['text'].apply(str)]
        tokenizer = AutoTokenizer.from_pretrained("Geotrend/bert-base-th-cased")
        bert_config = AutoConfig.from_pretrained("Geotrend/bert-base-th-cased", output_hidden_states=True)
        bert_model = TFAutoModel.from_pretrained('Geotrend/bert-base-th-cased', bert_config)
        
    else:
        Xo = [' '.join(process_text(item))  for item in df_ds['text'].apply(str)]
        tokenizer = BertTokenizer.from_pretrained(bert)
        bert_config = BertConfig.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
        bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased', bert_config)
        

    max_len = 45
    
    X_train, X_tmp, y_train, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=item, stratify=yo)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)

    num_class = np.unique(yo).shape[0]
    y_train_c = to_categorical(y_train)
    y_val_c = to_categorical(y_val)
    y_test_c = to_categorical(y_test)
    
    XT = X_train+X_val
    YT = np.vstack((y_train_c, y_val_c))
    
    bert_train = tokenize(XT, tokenizer)
    bert_test = tokenize(X_test, tokenizer)

    model = create_model(bert_model)
    
    recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    f1 = tfa.metrics.F1Score(num_classes=2, average='macro')
    mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
    auc = tf.keras.metrics.AUC()
    
    adam = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', recall, precision, mcc, f1, auc])
    model.summary()

    init(0)  
    ep = 60
    bs = 128
    #bs = int(len(X_train))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

    file = open(configs['output_scratch'] +"bert_10repeated_to.csv", "a")
    
    
    hist = model.fit(bert_train, YT, validation_data=(bert_test, y_test_c),
                      batch_size=bs, epochs=ep, verbose=1, callbacks=[es]) 
    
    file.write( str(item) + "," + str(max(hist.history['val_accuracy'])) + "," + str(max(hist.history['val_precision'])) + \
        "," + str(max(hist.history['val_recall'])) + ","  + str(max(hist.history['val_MatthewsCorrelationCoefficient'])) + \
            ","  + str(max(hist.history['val_auc'])) + "," + str(max(hist.history['val_f1_score'])))
    
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score, roc_auc_score)
    bert_pred = model.predict(bert_test)
    p = np.argmax(bert_pred, axis=-1)
    
    acc = accuracy_score(y_test, p)
    pre = precision_score(y_test,p, average='macro')
    rec = recall_score(y_test,p, average='macro')
    mcc = matthews_corrcoef(y_test,p)
    f1 = 2*pre*rec/(pre+rec)
    
    y_pred_bert =  np.zeros_like(bert_pred )
    y_pred_bert[np.arange(len(y_pred_bert)), bert_pred.argmax(1)] = 1
    

    #auc = roc_auc_score(y_test,y_pred_bert,multi_class='ovo',average='macro')
    auc = roc_auc_score(y_test,y_pred_bert[:,1])

    # test with test set
    # acc, pre, rec, mcc, auc, f1 = test_bert(clf, bert_test, y_test_c)
    file.write("," + str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1) + "\n")
    
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # sns.set()
    # plt.figure(num=None, figsize=(16, 8), dpi=90, facecolor='w', edgecolor='k')
    # plt.plot()
    # plt.plot(hist.history['accuracy'])
    # plt.plot(hist.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['Train', 'Val'], loc='upper left')
    # plt.show()
file.close()
    
