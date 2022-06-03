"""_supporting random embeddings and word2vec embedding_
"""
from enum import unique
from msilib import sequence
import sys
import pandas as pd 
import numpy as np 

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras import layers
from keras.layers.core import Reshape
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from matplotlib import pyplot
import src.utilities as utils
import src.feature.build_features as bf
from src.models.metrics import test, f1_m
import tensorflow as tf
import tensorflow_addons as tfa


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

EMBEDDING_DIM= 300
MAX_SEQUENCE_LENGTH = 500

config = utils.read_config()
df_ds = pd.read_csv(config['data']['processed_ws'])

iname = sys.argv[1]

y_ds = df_ds['target'].astype('category').cat.codes
f1 = tfa.metrics.F1Score(num_classes=df_ds['target'].unique().shape[0], average='macro')

Xo = df_ds['processed']
yo = y_ds.to_numpy()
dict = bf.get_dict_vocab()

def gensim_to_keras_embedding(model, train_embeddings):
    keyed_vectors = model.wv  
    weights = keyed_vectors.vectors  
    layer = layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=train_embeddings
    )
    return layer
        

file = open(config['output_scratch'] +"CNN_"+iname+ "_ws.csv", "a")
file.write("ACC, PRE, REC, MCC, AUC, F1 \n")

for item in range(9, 10):
    
    X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=item, stratify=yo)
    X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)


    tokenizer  = Tokenizer(num_words = MAX_SEQUENCE_LENGTH)
    tokenizer.fit_on_texts(X_train)
    train_sequences =  tokenizer.texts_to_sequences(X_train)
    valid_sequences = tokenizer.texts_to_sequences(X_val)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    vocab_size = len(tokenizer.word_index)
    print("vocab size is:", vocab_size)
    word_index = tokenizer.word_index
    
    w2v = Word2Vec.load(config['models'] + 'w2v_ws300.word2vec')

    vocabulary_size = min(len(word_index)+1, MAX_SEQUENCE_LENGTH)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    keyed_vectors = w2v.wv  
    for word, i in word_index.items():
        if i >= MAX_SEQUENCE_LENGTH:
            continue
        try:
            embedding_vector = keyed_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)



    from keras.layers import Embedding
    embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)

    # pad dataset to a maximum review length in words
    X_train_ps = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    X_val_ps = pad_sequences(valid_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    X_test_ps = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print(X_train_ps.shape, X_val_ps.shape, X_test_ps.shape)

    y_c = to_categorical(y)
    yv_c = to_categorical(yv)
    yt_c = to_categorical(yt)

    if iname == 'w2v':
        model = Sequential()
        model.add(gensim_to_keras_embedding(w2v, True))
        #model.add(layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
        
        model.add(layers.Dropout(0.5))
        model.add(layers.Conv1D(128, 3, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4, activation='softmax'))
    elif iname == 'w2v2d':
        
        sequence_length = X_train_ps.shape[1]
        encoder_input = Input(shape=(sequence_length,))
        emb = embedding_layer(encoder_input)
        reshape = Reshape((sequence_length, EMBEDDING_DIM, 1))(emb)
        conv = layers.Conv2D(128, 3, activation='relu')(reshape)
        max_pool_1 = layers.MaxPooling2D(2,2)(conv)
        conv_1 = layers.Conv2D(128, 3, activation='relu')(max_pool_1)
        max_pool_2 = layers.MaxPooling2D(2,2)(conv_1)
        flatten = layers.Flatten()(max_pool_2)
        drop = layers.Dropout(0.5)(flatten)
        output = layers.Dense(4, activation='softmax')(drop)
        model = Model(encoder_input, output)

        #model.build(input)
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy', f1])
    model.summary()

    # fit the model
    history = model.fit(X_train_ps, y_c,
                        epochs=30,
                        validation_data=(X_val_ps, yv_c),
                        batch_size=50)

    # evaluate model
    scores = model.evaluate(X_val_ps, yv_c, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    y_pred_prob = model.predict(X_val_ps)
    y_pred = np.argmax(model.predict(X_val_ps), axis=1)
    
    rounded_labels = np.argmax(yv_c, axis=1)
    acc_sc = accuracy_score(yv, y_pred)
    pre_sc = precision_score(yv, y_pred, average='macro')
    rec_sc = recall_score(yv, y_pred, average='macro')
    mcc_sc = matthews_corrcoef(yv, y_pred)
    f1_sc = 2*pre_sc*rec_sc/(pre_sc+rec_sc)
    auc_sc = roc_auc_score(yv, y_pred_prob, multi_class='ovo', average='macro')
    file.write(str(item) + "," +str(acc_sc) + "," + str(pre_sc) + "," + str(rec_sc) + "," + str(mcc_sc) + "," + str(auc_sc) + "," + str(f1_sc) + "\n")
    
    # test with unseen data
    # combine train and valid as a training set
    history = model.fit( np.vstack(X_train_ps, X_val_ps), np.hstack(y_c, yv_c), epochs=30, validation_data=(X_test_ps, yt_c), batch_size=50 )


# for investigation
print(classification_report(rounded_labels, y_pred))

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='valid')

pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='valid')
pyplot.legend()
#pyplot.show()
pyplot.savefig("cnn.png")