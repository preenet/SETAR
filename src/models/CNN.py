"""_supporting normal padding and word2vec embedding_
"""
import sys
import pandas as pd 
import numpy as np 

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from matplotlib import pyplot
import src.utilities as utils
import src.feature.build_features as bf
from src.models.metrics import test
import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

EMBEDDING_DIM= 300
MAX_SEQUENCE_LENGTH = 500

config = utils.read_config()
df_ds = pd.read_csv(config['data']['processed_ws'])

iname = sys.argv[1]

y_ds = df_ds['target'].astype('category').cat.codes

Xo = df_ds['processed']
yo = y_ds.to_numpy()
dict = bf.get_dict_vocab()

def gensim_to_keras_embedding(model, train_embeddings=False):
    keyed_vectors = model.wv  
    weights = keyed_vectors.vectors  
    layer = layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=train_embeddings,
    )
    return layer

if iname == 'w2v':
    w2v = Word2Vec(vector_size=EMBEDDING_DIM, min_count=1, window=4, workers=8)
    w2v.build_vocab(Xo)
    w2v.train(Xo, total_examples=w2v.corpus_count, epochs=100)

        
file = open(config['output_scratch'] +"CNN_"+iname+ "_ws.csv", "a")
file.write("ACC, PRE, REC, F1 \n")
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

    # pad dataset to a maximum review length in words
    X_train_ps = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    X_val_ps = pad_sequences(valid_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    X_test_ps = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print(X_train_ps.shape, X_val_ps.shape, X_test_ps.shape)

    y_c = to_categorical(y)
    yv_c = to_categorical(yv)
    yt_c = to_categorical(yt)

    model = Sequential()
    if iname == 'w2v':
        model.add(gensim_to_keras_embedding(w2v, True))
    else:
        model.add(layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax'))
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()

    # fit the model
    history = model.fit(X_train_ps, y_c,
                        epochs=2,
                        validation_data=(X_val_ps, yv_c),
                        batch_size=10)

    # evaluate model
    scores = model.evaluate(X_test_ps, yt_c, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    y_pred_prob = model.predict(X_test_ps)
    y_pred = np.argmax(model.predict(X_test_ps), axis=1)
    
    rounded_labels = np.argmax(yt_c, axis=1)

    acc_sc = accuracy_score(yt, y_pred)
    pre_sc = precision_score(yt, y_pred, average='macro')
    rec_sc = recall_score(yt, y_pred, average='macro')
    mcc_sc = matthews_corrcoef(yt, y_pred)
    f1_sc = f1_score(yt, y_pred, average='macro')
    auc_sc = roc_auc_score(yt, y_pred_prob, multi_class='ovo', average='macro')
    file.write(str(acc_sc) + "," + str(pre_sc) + "," + str(rec_sc) + "," + str(mcc_sc) + "," + str(auc_sc) + "," + str(f1_sc) + "\n")
file.close()

print(classification_report(rounded_labels, y_pred))

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
#pyplot.show()
pyplot.savefig("cnn.png")