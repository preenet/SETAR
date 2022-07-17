"""
Several feature extraction methods were applied on text feature to both corpuses as follows:  

* Bag of words supports both unigram and bigrams
* TF-IDF suppots both unigram and bigrams
* Word2Vec with TF-IDF embed vectorizer (300 dimension)
* POS_tagging with three methods using 17 Orchid tags and emoji tag
* Dictionary-based with lists of Thai positive and negative words for the unigram and bigram
  
Total of 8 text representations were extracted for each corpus.  
@Authors: pree.t@cmu.ac.th
"""
import sys

import numpy as np
import pandas as pd
import src.utilities as utils
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from pythainlp.tag import pos_tag_sents
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.feature.embedding_vectorizer import EmbeddingVectorizer
from src.feature.postag_transform import (flatten, onehot_label, tag, tag_emoj,
                                          word_tag)
from src.visualization.visualize import plot_top_feats, top_feats_all

plt.rcParams['font.family'] = 'tahoma'

# global vars
config = utils.read_config()
text_rep = ""
data_name = ""

__all__ = ['extract', 'get_dict_vocab']


def extract(text_rep: str, feat: pd.DataFrame, min_max: tuple):
    if text_rep == 'BOW':
        vect, feature = bow(feat, min_max)
    elif text_rep == 'TFIDF':
        vect, feature = tfidf(feat, min_max)
    elif text_rep == 'W2VTFIDF':
        vect, feature = w2vec(feat, 'tfidf', min_max)
    elif text_rep == 'W2VAVG':
        vect, feature = w2vec(feat, 'avg', min_max)
    elif text_rep == 'POSBOW':
        vect, feature = pos_bow(text_rep, feat, min_max)
    elif text_rep == 'POSBOWCONCAT':
        vect, feature = pos_bow(text_rep, feat, min_max)
    elif text_rep == 'POSBOWFLAT':
        vect, feature = pos_bow(text_rep, feat, min_max)
    elif text_rep == 'POSMEAN':
        return pos_mean_emb(feat)
    elif text_rep == 'POSW2V':
        vect, feature = pos_w2v_tfidf(feat, min_max)
    elif text_rep == 'DICTBOW':
        vect, feature = dict_bow(feat, min_max)
    elif text_rep == 'DICTTFIDF':
        vect, feature = dict_tfidf(feat, min_max)
    else:
        print("Error: invalid build feature method.")
        sys.exit(1)
    return vect, feature


def bow(feat: pd.DataFrame, min_max: tuple):
    print("Extracting BOW...")

    # lower min_df for bigram so sklearnex can run on the machine
    min_df = 40
    if min_max == (2, 2):
        min_df = 10
    bow = CountVectorizer(tokenizer=lambda x: x.split(), ngram_range=min_max, min_df=min_df)
    bow_fit = bow.fit(feat)
    text_bow = bow_fit.transform(feat)

    print(text_bow.toarray().shape)
    return bow_fit, text_bow


def tfidf(feat, min_max: tuple):
    print("Extracting TFI-IDF...")

    # lower min_df for bigram so sklearnex can run on the machine
    min_df = 20
    if min_max == (2, 2):
        min_df = 15
    tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(), ngram_range=min_max, min_df=min_df, sublinear_tf=True)
    tfidf_fit = tfidf.fit(feat)
    text_tfidf = tfidf_fit.transform(feat)

    print(text_tfidf.toarray().shape)

    return tfidf_fit, text_tfidf


def w2vec(feat: pd.DataFrame, type, min_max: tuple):
    print("Extracting W2V:", type, "...")
    tok_train = [text.split() for text in feat]
    w2v_model = Word2Vec(tok_train, vector_size=300, window=5, workers=4, min_count=1, seed=0, epochs=100)

    w2v_tfidf_emb = EmbeddingVectorizer(w2v_model, type, min_max)
    w2v_tifdf_fit = w2v_tfidf_emb.fit(feat)
    text_w2v_tfidf = w2v_tifdf_fit.transform(feat)

    return w2v_tifdf_fit, text_w2v_tfidf


def pos_bow(text_rep: str, feat: pd.DataFrame, min_max: tuple):
    print("Extracting ", text_rep + "...")
    tagged = pos_tag_sents(feat, corpus='orchid_ud')

    if text_rep == 'POSBOWCONCAT':
        pos = word_tag(tag_emoj(tagged))
        bow = CountVectorizer(ngram_range=min_max, min_df=20)
    elif text_rep == 'POSBOWFLAT':
        pos = flatten(tag_emoj(tagged))
        bow = CountVectorizer(ngram_range=min_max, min_df=20)
    elif text_rep == 'POSBOW':
        pos = tag(tag_emoj(tagged))
        bow = CountVectorizer(ngram_range=min_max)
    else:
        print("Error: incorrect type name.")
        sys.exit(1)

    pos_bow_fit = bow.fit(pos)
    text_pos_bow = pos_bow_fit.transform(pos)

    print(text_pos_bow.toarray().shape)

    return pos_bow_fit, text_pos_bow


def pos_mean_emb(feat: pd.DataFrame):
    print("Extracting POSMEAN...")
    tagged = pos_tag_sents(feat, corpus='orchid_ud')
    text_mean_emb = onehot_label(tag_emoj(tagged))
    return sparse.csr_matrix(text_mean_emb, dtype="float32")


def pos_w2v_tfidf(feat: pd.DataFrame, min_max: tuple):
    print("Extracting POSW2V TF-IDF...")
    tagged = pos_tag_sents(feat, corpus='orchid_ud')
    pos = word_tag(tag_emoj(tagged))
    # pos = [x.split(' ') for x in pos]

    w2v = Word2Vec(vector_size=300, min_count=1, window=4, workers=8, seed=0)
    w2v.build_vocab(pos)
    w2v.train(pos, total_examples=w2v.corpus_count, epochs=100)

    w2v_tfidf_emb = EmbeddingVectorizer(w2v, 'tfidf', min_max)
    w2v_tifdf_fit = w2v_tfidf_emb.fit(pos)
    text_w2v_tfidf = w2v_tifdf_fit.transform(pos)

    return w2v_tifdf_fit, text_w2v_tfidf


def dict_bow(feat: pd.DataFrame, min_max: tuple):
    print("Extracting DICT_BOW...")
    my_vocabs = get_dict_vocab()
    bow = CountVectorizer(vocabulary=my_vocabs, tokenizer=lambda x: x.split(), ngram_range=min_max)

    dict_bow_fit = bow.fit(feat)
    text_dict_bow = dict_bow_fit.transform(feat)

    print(text_dict_bow.toarray().shape)
    return dict_bow_fit, text_dict_bow


def dict_tfidf(feat: pd.DataFrame, min_max: tuple):
    print("Extracting DICT_TF-IDF...")
    my_vocabs = get_dict_vocab()
    tfidf1 = TfidfVectorizer(vocabulary=my_vocabs, tokenizer=lambda x: x.split(), ngram_range=min_max, min_df=20,
                             sublinear_tf=True)

    dict_tfidf_fit = tfidf1.fit(feat)
    text_dict_tfidf = dict_tfidf_fit.transform(feat)

    print(text_dict_tfidf.toarray().shape)
    return dict_tfidf_fit, text_dict_tfidf


def get_dict_vocab():
    # load list of our custom positive and negative words
    data_path_dict = config['data']['raw_dict']
    try:
        with open(data_path_dict + 'pos.txt', encoding='utf-8') as f:
            pos_words = [line.rstrip('\n') for line in f]
    except IOError as e:
        print("Error: can't open file to read", e)
        sys.exit(1)
    f.close()

    try:
        with open(data_path_dict + 'neg.txt', encoding='utf-8') as f:
            neg_words = [line.rstrip('\n') for line in f]

    except IOError as e:
        print("Error: can't open file to read", e)
        sys.exit(1)
    f.close()
    return np.unique(pos_words + neg_words)


def plot_feats(vectorizer, X, y):
    features = vectorizer.get_feature_names()
    tf = top_feats_all(X.toarray(), y, features)
    plot_top_feats(tf)
    return


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("*Error: incorrect number of arguments.")
        print("*Usage:[dataset name]", config['feature']['build_method'], "[min,max]")
        sys.exit(1)

    elif sys.argv[2] in config['feature']['build_method'] and sys.argv[1] in config['data']['name']:
        data_name = sys.argv[1]
        text_rep = sys.argv[2]
        min_max = sys.argv[3]

        print("Loading and converting from csv...")
        if data_name == 'kt':
            df_ds = pd.read_csv(config['data']['processed_kt'])
        elif data_name == 'ws':
            df_ds = pd.read_csv(config['data']['processed_ws'])
        elif data_name == 'tt':
            df_ds = pd.read_csv(config['data']['processed_tt'])
        else:
            sys.exit(1)
        print("*Building ", text_rep, "representation(s) for: ", data_name)
        extract(text_rep, df_ds['processed'], tuple(map(int, min_max.split(','))))
    else:
        print("*Error: incorrect argument name or dataset name.")
        sys.exit(1)
    print('*Program terminate successfully!')
