from pathlib import Path

import numpy as np
import pandas as pd
import src.utilities as utils
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from transformers import (BertConfig, BertTokenizer,
                          TFBertForSequenceClassification)

configs = utils.read_config()
root = utils.get_project_root()

df_ds = pd.read_csv(Path.joinpath(root, configs['data']['processed_ws']))

y_ds = df_ds['target'].astype('category').cat.codes
yo = y_ds.to_numpy()
Xo = df_ds['processed']


tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True)
tokenizer.fit_on_texts(Xo)
sequences_train_num = tokenizer.texts_to_sequences(Xo)
max_len = max([len(w) for w in sequences_train_num])
sequences_train_num = tf.keras.preprocessing.sequence.pad_sequences(sequences_train_num, maxlen=max_len )


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
def tokenize(sentences, tokenizer):
    input_ids, input_masks, input_segments = [],[],[]
    for sentence in sentences:
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, 
                                             return_attention_mask=True, return_token_type_ids=True)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])        
        
    return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32')


X_train, X_val, y_train, y_val = train_test_split(Xo, yo, test_size = 0.2)
num_class = np.unique(yo).shape[0]
y_train_c = to_categorical(y_train)
y_val_c = to_categorical(y_val)


bert_train = tokenize(X_train, tokenizer)
bert_val = tokenize(X_val, tokenizer)

# def create_model_direct():

    
#     return model_

def create_model_finetune():
    # Fine-tuning a Pretrained transformer model
    bert = 'bert-base-multilingual-cased'
    configuration = BertConfig.from_pretrained(bert, num_labels=64)

    configuration.output_hidden_states = True
    transformer_model = TFBertForSequenceClassification.from_pretrained(bert, config = configuration)

    input_ids_layer = tf.keras.layers.Input(shape=(max_len, ), dtype=np.int32)
    input_mask_layer = tf.keras.layers.Input(shape=(max_len ), dtype=np.int32)
    input_token_type_layer = tf.keras.layers.Input(shape=(max_len,), dtype=np.int32)

    bert_layer = transformer_model([input_ids_layer, input_mask_layer])[0]
   # flat_layer = tf.keras.layers.Flatten()(bert_layer)
    dropout= tf.keras.layers.Dropout(0.3)(bert_layer)
    dense_output = tf.keras.layers.Dense(num_class, activation='softmax')(dropout)

    model = tf.keras.Model(inputs=[input_ids_layer, input_mask_layer, input_token_type_layer ], outputs=dense_output)
    
    for layer in model.layers[:2]:
        layer.trainable = False
    return model

model = create_model_finetune()
model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

ep = 5
bs = 3
history = model.fit(bert_train, y_train_c, validation_data=(bert_val, y_val_c), batch_size=bs, epochs=ep) 
