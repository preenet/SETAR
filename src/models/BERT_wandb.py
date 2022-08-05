import sys
from pathlib import Path

import numpy as np
import pandas as pd
import src.utilities as utils
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from transformers import (BertConfig, BertTokenizer,
                          TFBertForSequenceClassification)
from wandb.keras import WandbCallback

import wandb


def main():
    configs = utils.read_config()
    root = utils.get_project_root()
    df_ds = pd.read_csv(Path.joinpath(root, configs['data']['processed_to']))

    y_ds = df_ds['target'].astype('category').cat.codes
    yo = y_ds.to_numpy()
    Xo = df_ds['processed']


    tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True)
    tokenizer.fit_on_texts(Xo)
    sequences_train_num = tokenizer.texts_to_sequences(Xo)
    max_len = max([len(w) for w in sequences_train_num])
    sequences_train_num = tf.keras.preprocessing.sequence.pad_sequences(sequences_train_num, maxlen=max_len )


    defaults = dict(
        dropout=0.5,
        learn_rate=0.0001,
        batch_size = 64,
        epochs=5
        )

    resume = sys.argv[-1] == "--resume"
    wandb.init(project="bert-kt", config=defaults, resume=resume, settings=wandb.Settings(_disable_stats=True))
    config = wandb.config

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


    X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=0, stratify=yo)
    X_val, X_test, yv, _ = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0, stratify=y_tmp)
    
    num_class = np.unique(y).shape[0]
    
    y_train_c = to_categorical(y)
    y_val_c = to_categorical(yv)
 

    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()
    recall = tf.keras.metrics.Recall()
    precision = tf.keras.metrics.Precision()
    auc = tf.keras.metrics.AUC()
    mcc = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=np.unique(y).shape[0])
    f1 = tfa.metrics.F1Score(num_classes=num_class, average='macro')

    bert_train = tokenize(X_train, tokenizer)
    bert_val = tokenize(X_val, tokenizer)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    
    def create_model_finetune():
        # Fine-tuning a Pretrained transformer model
        configuration = BertConfig.from_pretrained(bert)

        configuration.output_hidden_states = True
        transformer_model = TFBertForSequenceClassification.from_pretrained(bert, config = configuration)

        input_ids_layer = tf.keras.layers.Input(shape=(max_len, ), dtype=np.int32)
        input_mask_layer = tf.keras.layers.Input(shape=(max_len ), dtype=np.int32)
        
        embedding = transformer_model(input_ids_layer, input_mask_layer)[0]
        #dropout= tf.keras.layers.Dropout(config.dropout)(bert_layer)
        dense_output = tf.keras.layers.Dense(num_class, activation='softmax')(embedding)
        model = tf.keras.Model(inputs=[input_ids_layer, input_mask_layer], outputs=dense_output)
        
        # for layer in model.layers[:2]:
        #     layer.trainable = True
        return model

    model = create_model_finetune()                    
    model.compile(tf.keras.optimizers.Adam(lr=config.learn_rate), loss=loss, metrics=[accuracy, precision, recall, mcc, auc, f1])
    model.summary()

    model.fit(bert_train, y_train_c, validation_data=(bert_val, y_val_c), batch_size=config.batch_size, epochs=config.epochs,
              callbacks=[WandbCallback(save_model=True, monitor="loss"), es])
    return
    
if __name__ == "__main__":
    main()
