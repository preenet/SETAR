"""
This script extract features from three sentimental corpora: kt4.0 (data from our acquisition), thaitale, and wisesight. 
By training with our proposed SoTA classifiers on the kt4.0 dataset, 
we expect to see an improvement in the classification performance of the wisesight as well as thaitale dataset.

Several feature extraction methods were applied on text feature to both corpuses as follows:  

* Bag of words supports both unigram and bigrams
* TF-IDF suppots both unigram and bigrams
* Word2Vec with TF-IDF embbed vectorizer (300 dimension)
* POS_tagging using 17 Orchid tags and emoji tag
* Dictionary-based with a custom list of Thai positive and negative words for the unigram and bigrams
  
Total of 8 text representations were exctracted for each corpus.  
@Authors: pree.t@cmu.ac.th
"""
import os, sys
import pandas as pd
import numpy as np
import joblib
import src.utilities as utils
import logging

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from scipy import sparse # for converting w2v embbed vector to spare matrix and targets  

from src.feature.tfidf_embedding_vectorizer import TfidfEmbeddingVectorizer
from src.feature.process_thai_text import process_text
from src.feature.pos_rule import word_tag, tag, tag_emoj
from src.visualization.visualize import top_feats_all, plot_top_feats
from pythainlp.tag import pos_tag_sents
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'tahoma'


# get config file
config = utils.read_config()

# output folder to be put in
out_path = os.path.join(config['output'])


def extract(method):
    df_kt = pd.read_csv(config['data']['raw_kt'])
    df_ws = pd.read_csv(config['data']['raw_ws'])

    y_kt = df_kt['vote'].astype('category').cat.codes
    y_ws = df_ws['targets'].astype('category').cat.codes
    print("label kt:", y_kt.unique(), ", label ws", y_ws.unique())

    df_kt['processed'] = df_kt['text'].apply(str).apply(process_text)
    df_ws['processed'] = df_ws['texts'].apply(str).apply(process_text)

    logging.info(df_ws[0:10])
    logging.info(df_kt.head(10))
    logging.info(df_kt.describe())
    logging.info(df_ws.tail(10))
    logging.info(df_ws.describe())

    # class distribution
    logging.info("KT clssses dist:", df_kt.vote.value_counts() / df_kt.shape[0])

    # class distribution
    logging.info("WS classes dist:", df_ws.targets.value_counts() / df_ws.shape[0])

    y_t_kt = y_kt.to_numpy().reshape(-1, 1)
    y_t_ws = y_ws.to_numpy().reshape(-1, 1)

    y_t_kt = sparse.csr_matrix(y_t_kt)
    y_t_ws = sparse.csr_matrix(y_t_ws)

    if(method == 'BOW'):
        logging.info("Extracting BOW")            
        bow(df_kt, y_t_kt, df_ws, y_t_ws)

    elif(method == 'TFIDF'):
        logging.info("Extracting TFI-IDF")  
        tfidf(df_kt, y_t_kt, df_ws, y_t_ws)

    elif(method == 'W2V'):
        logging.info("Extracting W2V-TFIDF")  
        w2v_tfidf(df_kt, y_t_kt, df_ws, y_t_ws)

    elif(method == 'POSBOW'):
        logging.info("Extracting POS_BOW")   
        pos_bow(df_kt, y_t_kt, df_ws, y_t_ws)

    elif(method == 'POSTFIDF'):
        logging.info("Extracting POS_TF-IDF")  
        pos_tfidf(df_kt, df_ws)

    elif(method == 'DICTBOW'):
        my_vocabs = get_dict_vocab()
        logging.info("Extracting DICT_BOW")  
        dict_bow(df_kt, y_t_kt, df_ws, y_t_ws)

    elif(method == 'DICTTFIDF'):
        logging.info("Extracting DICT_TF-IDF")  
        dict_tfidf(df_kt, df_ws, my_vocabs)

    elif(method == 'ALL'):
        logging.info("Extracted with all methods.")
        bow(df_kt, y_t_kt, df_ws, y_t_ws)
        tfidf(df_kt, y_t_kt, df_ws, y_t_ws)
        w2v_tfidf(df_kt, y_t_kt, df_ws, y_t_ws)
        pos_bow(df_kt, y_t_kt, df_ws, y_t_ws)
        pos_tfidf(df_kt, df_ws)
        dict_bow(df_kt, y_t_kt, df_ws, y_t_ws)
        dict_tfidf(df_kt, df_ws, my_vocabs)
    else:
        sys.exit(1)

    return

def plot_feats(vectorizer, X, y):
    features = vectorizer.get_feature_names()
    tf = top_feats_all(X.toarray(), y, features)
    plot_top_feats(tf)
    return
    
def bow(df_kt, y_t_kt, df_ws, y_t_ws):
   # BOW with unigram and bigrams
    bow1 = CountVectorizer(tokenizer=process_text, ngram_range=(1, 1), min_df=5)
    bow2 = CountVectorizer(tokenizer=process_text, ngram_range=(2, 2), min_df=5)

    # fit kt and transform to both datasets
    bow1_fit_kt = bow1.fit(df_kt['text'].apply(str))
    text_bow1_kt = bow1_fit_kt.transform(df_kt['text'].apply(str))
    lex_bow1_kt = bow1_fit_kt.get_feature_names()
    text_bow1_ws = bow1_fit_kt.transform(df_ws['texts'].apply(str))
    lex_bow1_ws = bow1_fit_kt.get_feature_names()

    bow2_fit_kt = bow2.fit(df_kt['text'].apply(str))
    text_bow2_kt = bow2_fit_kt.transform(df_kt['text'].apply(str))
    lex_bow2_kt = bow2_fit_kt.get_feature_names()
    text_bow2_ws = bow2_fit_kt.transform(df_ws['texts'].apply(str))
    lex_bow2_ws = bow2_fit_kt.get_feature_names()

    logging.debug(text_bow1_kt.toarray().shape,  text_bow1_kt.toarray().shape)
    logging.debug(text_bow2_kt.toarray().shape,  text_bow2_kt.toarray().shape, end = " ")
    logging.debug(text_bow1_ws.toarray().shape,  text_bow1_ws.toarray().shape)
    logging.debug(text_bow2_ws.toarray().shape,  text_bow2_ws.toarray().shape, end = " ")
    
    logging.debug(len(lex_bow1_kt), len(lex_bow1_ws))
    logging.debug(len(lex_bow2_kt), len(lex_bow2_ws))

    # visualize 
    y = y_t_kt.todense()
    y = np.array(y.reshape(y.shape[0],))[0]
    plot_feats(bow1_fit_kt, text_bow1_kt, y)

    # write to disk
    arr_bow1_kt = np.hstack((text_bow1_kt, y_t_kt))
    arr_bow2_kt = np.hstack((text_bow2_kt, y_t_kt))
    joblib.dump(arr_bow1_kt, config['output']+'text_bow1_kt.pkl')
    joblib.dump(arr_bow2_kt, config['output']+'text_bow2_kt.pkl')

    joblib.dump(lex_bow1_kt, config['output']+'lex_bow1_kt.pkl')
    joblib.dump(lex_bow2_kt, config['output']+'lex_bow2_kt.pkl')

    arr_bow1_ws = np.hstack((text_bow1_ws, y_t_ws))
    arr_bow2_ws = np.hstack((text_bow2_ws, y_t_ws))
    joblib.dump(arr_bow1_ws, config['output']+'text_bow1_ws.pkl')
    joblib.dump(arr_bow2_ws, config['output']+'text_bow2_ws.pkl')

    joblib.dump(lex_bow1_ws, config['output']+'lex_bow1_ws.pkl')
    joblib.dump(lex_bow2_ws, config['output']+'lex_bow2_ws.pkl')
    return 


def tfidf(df_kt, y_t_kt, df_ws, y_t_ws):
    # TF-IDF with unigram and bigrams
    tfidf1 = TfidfVectorizer(tokenizer=process_text, ngram_range=(1, 1), min_df=5)
    tfidf2 = TfidfVectorizer(tokenizer=process_text, ngram_range=(2, 2), min_df=5)

    # fit kt and transform to both datasets
    tfidf1_fit_kt = tfidf1.fit(df_kt['text'].apply(str))
    text_tfidf1_kt = tfidf1_fit_kt.transform(df_kt['text'].apply(str))
    lex_tfidf1_kt = tfidf1_fit_kt.get_feature_names()
    text_tfidf1_ws = tfidf1_fit_kt.transform(df_ws['texts'].apply(str))
    lex_tfidf1_ws = tfidf1_fit_kt.get_feature_names()

    tfidf2_fit_kt = tfidf2.fit(df_kt['text'].apply(str))
    text_tfidf2_kt = tfidf2_fit_kt.transform(df_kt['text'].apply(str))
    lex_tfidf2_kt = tfidf2_fit_kt.get_feature_names()
    text_tfidf2_ws = tfidf2_fit_kt.transform(df_ws['texts'].apply(str))
    lex_tfidf2_ws = tfidf2_fit_kt.get_feature_names()

    logging.debug(text_tfidf1_kt.toarray().shape,  text_tfidf1_kt.toarray().shape)
    logging.debug(text_tfidf2_kt.toarray().shape,  text_tfidf2_kt.toarray().shape, end =" ")
    logging.debug(text_tfidf1_ws.toarray().shape,  text_tfidf1_ws.toarray().shape)
    logging.debug(text_tfidf2_ws.toarray().shape,  text_tfidf2_ws.toarray().shape, end =" ")

    print(len(lex_tfidf1_kt), len(lex_tfidf1_ws))
    print(len(lex_tfidf2_kt), len(lex_tfidf2_ws))

    arr_tfidf1_kt = np.hstack((text_tfidf1_kt, y_t_kt))
    arr_tfidf2_kt = np.hstack((text_tfidf2_kt, y_t_kt))
    joblib.dump(arr_tfidf1_kt, config['output']+'text_tfidf1_kt.pkl')
    joblib.dump(arr_tfidf2_kt, config['output']+'text_tfidf2_kt.pkl')

    joblib.dump(lex_tfidf1_kt, config['output']+'lex_tfidf1_kt.pkl')
    joblib.dump(lex_tfidf2_kt, config['output']+'lex_tfidf2_kt.pkl')
    return


def w2v_tfidf(df_kt, y_t_kt, df_ws, y_t_ws):
    # create word2vec for kt corpus
    w2v_kt = Word2Vec(vector_size=300, min_count=1, window=4, workers=4)
    w2v_kt.build_vocab(df_kt['processed'])
    w2v_kt.train(df_kt['processed'], total_examples=w2v_kt.corpus_count, epochs=100)
    print(w2v_kt.wv.most_similar("บะหมี่"))

    w2v_tfidf_emb_kt = TfidfEmbeddingVectorizer(w2v_kt)
    w2v_tifdf_fit_kt = w2v_tfidf_emb_kt.fit(df_kt['processed'].apply(str))

    # transfrom on both corpuses
    text_w2v_tfidf_kt = w2v_tifdf_fit_kt.transform(df_kt['processed'])
    text_w2v_tfidf_ws = w2v_tifdf_fit_kt.transform(df_ws['processed'])

    arr_w2v_tfidf_kt = np.hstack(( sparse.csr_matrix(text_w2v_tfidf_kt), y_t_kt))
    arr_w2v_tfidf_ws = np.hstack(( sparse.csr_matrix(text_w2v_tfidf_ws), y_t_ws))
    joblib.dump(arr_w2v_tfidf_kt, config['output']+'text_w2v_tfidf_kt.pkl')
    joblib.dump(arr_w2v_tfidf_ws, config['output']+'text_w2v_tfidf_ws.pkl')
    return 

def pos_bow(df_kt, y_t_kt, df_ws, y_t_ws):

    pos = pos_tag_sents(df_kt['processed'].tolist(), corpus='orchid_ud')
    pos = tag_emoj(pos)
    df_kt['post_tag1'] = pd.DataFrame(tag(pos))
    df_kt['post_tag2'] = pd.DataFrame(word_tag(pos))

    pos = pos_tag_sents(df_ws['processed'].tolist(), corpus='orchid_ud')
    pos = tag_emoj(pos)
    df_ws['post_tag1'] = pd.DataFrame(tag(pos))
    df_ws['post_tag2'] = pd.DataFrame(word_tag(pos))

    logging.debug(df_ws[['processed', 'post_tag1']].iloc[1000:1010])

    logging.debug(df_ws['post_tag2'].iloc[1000:1010])

    # create bow vectors

    bow1 = CountVectorizer(ngram_range=(1, 1))
    #bow2 = CountVectorizer(ngram_range=(2, 2))

    text_pos_bow1_fit_kt = bow1.fit(df_kt['post_tag1'])
    text_pos_bow1_kt = text_pos_bow1_fit_kt.transform(df_kt['post_tag1'])
    text_pos_bow1_ws = text_pos_bow1_fit_kt.transform(df_ws['post_tag1'])

    # text_pos_bow2_fit_kt = bow2.fit(df_kt['post_tag1'])
    # text_pos_bow2_kt = text_pos_bow2_fit_kt.transform(df_kt['post_tag2'])
    # text_pos_bow2_ws = text_pos_bow2_fit_kt.transform(df_ws['post_tag2'])

    logging.debug(text_pos_bow1_kt.toarray().shape,  text_pos_bow1_kt.toarray().shape)
    logging.debug(text_pos_bow1_ws.toarray().shape,  text_pos_bow1_ws.toarray().shape)

    # print(text_pos_bow2_kt.toarray().shape,  text_pos_bow2_kt.toarray().shape)
    # print(text_pos_bow2_ws.toarray().shape,  text_pos_bow2_ws.toarray().shape)

    arr_pos_bow1_kt = np.hstack((text_pos_bow1_kt, y_t_kt))
    #arr_pos_bow2_kt = np.hstack((text_pos_bow2_kt, y_t_kt))
    joblib.dump(arr_pos_bow1_kt, config['output']+'text_pos_bow1_kt.pkl')
    #joblib.dump(arr_pos_bow2_kt, config['output']+'text_pos_bow2_kt.pkl')

    arr_pos_bow1_ws = np.hstack((text_pos_bow1_ws, y_t_ws))
    #arr_pos_bow2_ws = np.hstack((text_pos_bow2_ws, y_t_ws))
    joblib.dump(arr_pos_bow1_ws, config['output']+'text_pos_bow1_ws.pkl')
    #joblib.dump(arr_pos_bow2_ws, config['output']+'text_pos_bow2_ws.pkl')
    return

def pos_tfidf(df_kt, y_t_kt, df_ws, y_t_ws):
    # create tfidf vectors
    tfidf1 = TfidfVectorizer(ngram_range=(1, 1))
    tfidf2 = TfidfVectorizer(ngram_range=(2, 2))

    text_pos_tfidf1_fit_kt = tfidf1.fit(df_kt['post_tag1'])
    text_pos_tfidf1_kt = text_pos_tfidf1_fit_kt.transform(df_kt['post_tag1'])
    text_pos_tfidf1_ws = text_pos_tfidf1_fit_kt.transform(df_ws['post_tag1'])

    text_pos_tfidf2_fit_kt = tfidf2.fit(df_kt['post_tag1'])
    text_pos_tfidf2_kt = text_pos_tfidf2_fit_kt.transform(df_kt['post_tag2'])
    text_pos_tfidf2_ws = text_pos_tfidf2_fit_kt.transform(df_ws['post_tag2'])

    logging.debug(text_pos_tfidf1_kt.toarray().shape,  text_pos_tfidf1_kt.toarray().shape)
    logging.debug(text_pos_tfidf1_ws.toarray().shape,  text_pos_tfidf1_ws.toarray().shape)

    logging.debug(text_pos_tfidf2_kt.toarray().shape,  text_pos_tfidf2_kt.toarray().shape)
    logging.debug(text_pos_tfidf2_ws.toarray().shape,  text_pos_tfidf2_ws.toarray().shape)

    # arr_pos_tfidf1_kt = np.hstack((text_pos_tfidf1_kt, y_t_kt))
    # arr_pos_tfidf2_kt = np.hstack((text_pos_tfidf2_kt, y_t_kt))
    # joblib.dump(arr_pos_tfidf1_kt, config['output']+'text_pos_tfidf1_kt.pkl')
    # joblib.dump(arr_pos_tfidf2_kt, config['output']+'text_pos_tfidf2_kt.pkl')

    # arr_pos_tfidf1_ws = np.hstack((text_pos_tfidf1_ws, y_t_ws))
    # arr_pos_tfidf2_ws = np.hstack((text_pos_tfidf2_ws, y_t_ws))
    # joblib.dump(arr_pos_tfidf1_ws, config['output']+'text_pos_tfidf1_ws.pkl')
    # joblib.dump(arr_pos_tfidf2_ws, config['output']+'text_pos_tfidf2_ws.pkl')
    return


def get_dict_vocab():
    # load list of our custom positive and negative words
    data_path_dict = config['data']['raw_dict']
    with open(data_path_dict + 'pos_words.txt', encoding='utf-8') as f:
        pos_words = [line.rstrip('\n') for line in f]

    with open(data_path_dict + 'neg_words.txt', encoding='utf-8') as f:
        neg_words = [line.rstrip('\n') for line in f]
        pos_words = list(set(pos_words))
        neg_words = list(set(neg_words))

    my_vocabs = pos_words + neg_words
    logging.debug('dict size: ', len(my_vocabs))

    return my_vocabs

def dict_bow(df_kt, y_t_kt, df_ws, y_t_ws):
    my_vocabs = get_dict_vocab()
    bow1 = CountVectorizer(tokenizer=process_text, ngram_range=(1, 1))
    bow2 = CountVectorizer(tokenizer=process_text, ngram_range=(2, 2))

    text_dict_bow1_fit = bow1.fit(my_vocabs)
    text_dict_bow1_kt = text_dict_bow1_fit.transform(df_kt['text'].apply(str))
    text_dict_bow1_ws = text_dict_bow1_fit.transform(df_ws['texts'].apply(str))

    text_dict_bow2_fit = bow2.fit(my_vocabs)
    text_dict_bow2_kt = text_dict_bow2_fit.transform(df_kt['text'].apply(str))
    text_dict_bow2_ws = text_dict_bow2_fit.transform(df_ws['texts'].apply(str))

    logging.debug(text_dict_bow1_kt.toarray().shape,  text_dict_bow1_kt.toarray().shape)
    logging.debug(text_dict_bow1_ws.toarray().shape,  text_dict_bow1_ws.toarray().shape)

    logging.debug(text_dict_bow2_kt.toarray().shape,  text_dict_bow2_kt.toarray().shape)
    logging.debug(text_dict_bow2_ws.toarray().shape,  text_dict_bow2_ws.toarray().shape)

    arr_dict_bow1_kt = np.hstack((text_dict_bow1_kt, y_t_kt))
    arr_dict_bow2_kt = np.hstack((text_dict_bow2_kt, y_t_kt))
    joblib.dump(arr_dict_bow1_kt, config['output']+'text_dict_bow1_kt.pkl')
    joblib.dump(arr_dict_bow2_kt, config['output']+'text_dict_bow2_kt.pkl')

    arr_dict_bow1_ws = np.hstack((text_dict_bow1_ws, y_t_ws))
    arr_dict_bow2_ws = np.hstack((text_dict_bow2_ws, y_t_ws))
    joblib.dump(arr_dict_bow1_ws, config['output']+'text_dict_bow1_ws.pkl')
    joblib.dump(arr_dict_bow2_ws, config['output']+'text_dict_bow2_ws.pkl')
    return

def dict_tfidf(df_kt, y_t_kt, df_ws, y_t_ws):
    my_vocabs = get_dict_vocab()
    tfidf1 = TfidfVectorizer(tokenizer=process_text, ngram_range=(1, 1))
    tfidf2 = TfidfVectorizer(tokenizer=process_text, ngram_range=(2, 2))

    text_dict_tfidf1_fit = tfidf1.fit(my_vocabs)
    text_dict_tfidf1_kt = text_dict_tfidf1_fit.transform(df_kt['text'].apply(str))
    text_dict_tfidf1_ws = text_dict_tfidf1_fit.transform(df_ws['texts'].apply(str))

    text_dict_tfidf2_fit = tfidf2.fit(my_vocabs)
    text_dict_tfidf2_kt = text_dict_tfidf2_fit.transform(df_kt['text'].apply(str))
    text_dict_tfidf2_ws = text_dict_tfidf2_fit.transform(df_ws['texts'].apply(str))

    logging.debug(text_dict_tfidf1_kt.toarray().shape,  text_dict_tfidf1_kt.toarray().shape)
    logging.debug(text_dict_tfidf1_ws.toarray().shape,  text_dict_tfidf1_ws.toarray().shape)

    logging.debug(text_dict_tfidf2_kt.toarray().shape,  text_dict_tfidf2_kt.toarray().shape)
    logging.debug(text_dict_tfidf2_ws.toarray().shape,  text_dict_tfidf2_ws.toarray().shape)

    arr_dict_tfidf1_kt = np.hstack((text_dict_tfidf1_kt, y_t_kt))
    arr_dict_tfidf2_kt = np.hstack((text_dict_tfidf2_kt, y_t_kt))
    joblib.dump(arr_dict_tfidf1_kt, config['output']+'text_dict_tfidf1_kt.pkl')
    joblib.dump(arr_dict_tfidf2_kt, config['output']+'text_dict_tfidf2_kt.pkl')

    arr_dict_tfidf1_ws = np.hstack((text_dict_tfidf1_ws, y_t_ws))
    arr_dict_tfidf2_ws = np.hstack((text_dict_tfidf2_ws, y_t_ws))
    joblib.dump(arr_dict_tfidf1_ws, config['output']+'text_dict_tfidf1_ws.pkl')
    joblib.dump(arr_dict_tfidf2_ws, config['output']+'text_dict_tfidf2_ws.pkl')
    return


if __name__ == "__main__":
    logging.basicConfig(filename=config['output']+'build_features.log', level=logging.INFO)
    
    if (len(sys.argv) != 2):
        logging.error("ERROR: need argument")
        logging.info(config['feature']['build_method'])
        sys.exit(1)

    elif sys.argv[1] in config['feature']['build_method']: 
        logging.info(sys.argv[1], "Building text representations...")
        extract(sys.argv[1])
        logging.info("Finished building features!")
    else:
        logging.error("ERROR: feature extraction method doesn't support.")
        sys.exit(1)
    logging.info('Program terminate sucessfully!')