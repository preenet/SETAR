import sys
import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers, Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from matplotlib import pyplot
import src.utilities as utils
import src.feature.build_features as bf
from src.models.metrics import test_deep
import tensorflow as tf
import tensorflow_addons as tfa
from gensim.models import Word2Vec

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

config = utils.read_config()
df_ds = pd.read_csv(config['data']['processed_ws'])

iname = sys.argv[1]

y_ds = df_ds['target'].astype('category').cat.codes

Xo = df_ds['processed']
yo = y_ds.to_numpy()

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

file = open(config['output_scratch'] +"BiLSTM_"+iname+ "_ws.csv", "a")

for item in range(9, 10):
    
    X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=item, stratify=yo)
    X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)
    
    f1 = tfa.metrics.F1Score(num_classes=np.unique(y).shape[0], average='macro')

    MAX_SEQUENCE_LENGTH = 500
    EMBEDDING_DIM= 300

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
    
    # from pythainlp import word_vector
    # w2v = Word2Vec(vector_size=300, min_count=1, window = 5, workers=8)
    # w2v.build_vocab(Xo)
    # w2v_thwiki = word_vector.get_model()
    # w2v.build_vocab(w2v_thwiki.index_to_key, update=True)
    # w2v.wv.vectors_lockf = np.ones(len(w2v.wv))
    # w2v.wv.intersect_word2vec_format(config['models']+'thai2vec.bin', binary=True, lockf=1.0)
    
    # w2v.train(Xo, total_examples=w2v.corpus_count, epochs=300)
    w2v = Word2Vec.load(config['models'] + 'w2v_ws_thwiki300.word2vec')
    
    embedding_matrix_ft = np.random.random((len(tokenizer.word_index)+1, w2v.vector_size))
    pas = 0
    for word,i in tokenizer.word_index.items():
        try:
            embedding_matrix_ft[i] = w2v.wv[word]
        except:
            pas+=1

    model = Sequential()
    # 

    # #model.add(layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    # #model.add(layers.Bidirectional(layers.LSTM(256)))
    # model.add(layers.Bidirectional(layers.LSTM(300, return_sequences=True, dropout=0.5), merge_mode ='concat'))
    # model.add(layers.TimeDistributed(layers.Dense(300,  activation='relu')))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(4, activation='softmax'))
    
    vocab_size, emdedding_size = embedding_matrix_ft.shape
    model = Sequential()
    #model.add(layers.Embedding(input_dim=vocab_size, 
                                                    # output_dim=emdedding_size, 
                                                    # weights=[embedding_matrix_ft],
                                                    # ))
    #model.add(layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)) 
    model.add(gensim_to_keras_embedding(w2v, True)) # use gensim word2vec
    
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(layers.LSTM(300, dropout=0.5), merge_mode='concat'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax'))
    
    # input = layers.Input(shape=(MAX_SEQUENCE_LENGTH,))
    # model = gensim_to_keras_embedding(w2v, True)(input)
    # model = layers.Dropout(0.5)(model)
    # model = layers.Bidirectional(layers.LSTM(300, return_sequences=True, dropout=0.5), merge_mode ='concat')(model)
    # model = layers.TimeDistributed(layers.Dense(300,  activation='relu'))(model)
    # model = layers.Dropout(0.5)(model)
    # model = layers.Flatten()(model)
    # model = layers.Dense(100, activation='relu')(model)
    # output = layers.Dense(4, activation='softmax')(model)
    # model = Model(input, output)
    
    # sequence_input = layers.Input(shape=(MAX_SEQUENCE_LENGTH, ))
    # x = gensim_to_keras_embedding(w2v, False)(sequence_input)
    # x = layers.SpatialDropout1D(0.2)(x)
    # x = layers.Bidirectional(layers.GRU(64, return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
    # x = layers.Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
    # avg_pool = layers.GlobalAveragePooling1D()(x)
    # max_pool = layers.GlobalMaxPooling1D()(x)
    # x = layers.concatenate([avg_pool, max_pool]) 
    
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy', f1])
    model.summary()

    # fit the model
    history = model.fit(X_train_ps, y_c,
                        epochs=30,
                        validation_data=(X_val_ps, yv_c),
                        batch_size=64)

    # evaluate model
    scores = model.evaluate(X_val_ps, yv_c, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    acc, pre, rec, mcc, auc, f1 = test_deep(model, X_val_ps, yv)
    file.write(str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1))
    
    # combine train and valid as a training set, train a model and test with unseen data
    #history = model.fit( np.vstack((X_train_ps, X_val_ps)), np.vstack((y_c, yv_c)), epochs=30, validation_data=(X_test_ps, yt_c), batch_size=50 )
    #acc, sens, spec, mcc, roc, f1 = test_deep(model, X_test_ps, yt)
    #file.write("," + str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1) + "\n")
file.close()

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
pyplot.savefig("bi-lstm.png")