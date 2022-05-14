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
import pandas as pd
import numpy as np
from sklearn.preprocessing import maxabs_scale
import src.utilities as utils

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pythainlp.tag import pos_tag_sents
from gensim.models import Word2Vec

from src.feature.tfidf_embedding_vectorizer import TfidfEmbeddingVectorizer

from src.feature.pos_rule import word_tag, tag, tag_emoj
from src.visualization.visualize import top_feats_all, plot_top_feats

from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'tahoma'

# global vars
config = utils.read_config()
text_rep = ""
data_name = ""

__all__ = ['extract']

def extract(df_ds, min_max):

    # transfrom target string to int
    y = df_ds['target'].astype('category').cat.codes
    print("label: ", y.unique())

    print("Quick peek for downstream task:")
    print(df_ds[0:10])
    print(df_ds.tail(10))
    print(df_ds.describe())

    # class distribution
    print("DS classes dist.:", df_ds.target.value_counts() / df_ds.shape[0])

    if text_rep == 'BOW':        
        vect, feature = bow(df_ds, min_max)
    elif text_rep == 'TFIDF':    
        vect, feature = tfidf(df_ds, min_max)
    elif text_rep == 'W2V':    
        vect, feature = w2v_tfidf(df_ds, min_max)
    elif text_rep == 'POSBOW': 
        vect, feature = pos_bow(df_ds, min_max)
    elif(text_rep == 'POSTFIDF'):
        vect, feature = pos_tfidf(df_ds, min_max)
    elif(text_rep == 'DICTBOW'):
        vect, feature = dict_bow(df_ds, min_max)
    elif(text_rep == 'DICTTFIDF'):
        vect, feature = dict_tfidf(df_ds, min_max)
    elif(text_rep== 'ALL'):
        print("Extracted with all methods...")
        bow(df_ds, min_max)
        tfidf(df_ds, min_max)
        w2v_tfidf(df_ds)
        pos_bow(df_ds, min_max)
        pos_tfidf(df_ds, min_max)
        dict_bow(df_ds, min_max)
        dict_tfidf(df_ds, min_max)
    else:
        sys.exit(1)
    return vect, feature

def bow(df_ds, min_max):
    print("Extracting BOW...")  
    bow = CountVectorizer(tokenizer=lambda x:x.split(), ngram_range=min_max, min_df=5)

    # fit kt and transform to both datasets
    bow_fit = bow.fit(df_ds['processed'].apply(str))
    text_bow = bow_fit.transform(df_ds['processed'].apply(str))

    print(text_bow.toarray().shape)
    
    # visualize 
    # y = yt.todense()
    # y = np.array(y.reshape(y.shape[0],))[0]
    # plot_feats(bow1_fit, text_bow1, y)

    return bow_fit, text_bow

def tfidf(df_ds, min_max):
    print("Extracting TFI-IDF...")  
    tfidf = TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=min_max, min_df=5)

    # fit kt and transform to both datasets
    tfidf_fit = tfidf.fit(df_ds['processed'].apply(str))
    text_tfidf = tfidf_fit.transform(df_ds['processed'].apply(str))
    
    print(text_tfidf.toarray().shape)

    return tfidf_fit, text_tfidf

def w2v_tfidf(df_ds):
    print("Extracting W2V-TFIDF...")  
    # create word2vec for kt corpus
    w2v = Word2Vec(vector_size=300, min_count=1, window=4, workers=4)
    w2v.build_vocab(df_ds['processed'])
    w2v.train(df_ds['processed'], total_examples=w2v.corpus_count, epochs=100)

    w2v_tfidf_emb = TfidfEmbeddingVectorizer(w2v)
    w2v_tifdf_fit = w2v_tfidf_emb.fit(df_ds['processed'])
    text_w2v_tfidf = w2v_tifdf_fit.transform(df_ds['processed'])

    return w2v_tifdf_fit, text_w2v_tfidf

def pos_bow(df_ds, min_max):
    print("Extracting POS_BOW...")   

    pos = pos_tag_sents(df_ds['processed'].tolist(), corpus='orchid_ud')
    pos = tag_emoj(pos)
    df_ds['pos_tag1'] = pd.DataFrame(tag(pos))
    df_ds['pos_tag2'] = pd.DataFrame(word_tag(pos))

    print(df_ds[['processed', 'pos_tag1']].iloc[1000:1010])
    print(df_ds['pos_tag2'].iloc[1000:1010])

    # create bow vectors
    bow = CountVectorizer(ngram_range=min_max)

    pos_bow_fit = bow.fit(df_ds['pos_tag1'])
    text_pos_bow = pos_bow_fit.transform(df_ds['pos_tag1'])

    print(text_pos_bow.toarray().shape)

    return pos_bow_fit, text_pos_bow

def pos_tfidf(df_ds, min_max):
    print("Extracting POS_TF-IDF...")  
    tfidf = TfidfVectorizer(ngram_range=min_max)

    pos_tfidf_fit = tfidf.fit(df_ds['pos_tag1'])
    text_pos_tfidf = pos_tfidf_fit.transform(df_ds['pos_tag1'])

    print(text_pos_tfidf.toarray().shape)

    return pos_tfidf_fit, text_pos_tfidf

def dict_bow(df_ds, min_max):
    print("Extracting DICT_BOW...")  
    my_vocabs = get_dict_vocab()
    bow = CountVectorizer(vocabulary=my_vocabs, tokenizer=lambda x:x.split(), ngram_range=min_max)

    dict_bow_fit = bow.fit(df_ds['processed'].apply(str))
    text_dict_bow = dict_bow_fit.transform(df_ds['processed'].apply(str))

    print(text_dict_bow.toarray().shape)
    return dict_bow_fit, text_dict_bow

def dict_tfidf(df_ds, min_max):
    print("Extracting DICT_TF-IDF...")  
    my_vocabs = get_dict_vocab()
    tfidf1 = TfidfVectorizer(vocabulary=my_vocabs, tokenizer=lambda x:x.split(), ngram_range=min_max)

    dict_tfidf_fit = tfidf1.fit(df_ds['processed'].apply(str))
    text_dict_tfidf = dict_tfidf_fit.transform(df_ds['processed'].apply(str))
    
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

# def write_to_disk(text_rep, y, file_name):
#     X_train, X_tmp, y_train, y_tmp = train_test_split(text_rep, y, test_size=0.4, random_state=0, stratify=y)
#     X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0, stratify=y_tmp)
    
#     joblib.dump(np.hstack((X_train, y_train)), config['data']['final_train_val_test'] + "train_" + file_name)
#     joblib.dump(np.hstack((X_val, y_val)), config['data']['final_train_val_test'] + "val_" + file_name)
#     joblib.dup(np.hstack((X_test, y_test)), config['data']['final_train_val_test']+ "test_" + file_name)
#     return

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
            df_ds = pd.read_csv(config['data']['processed_kt'], converters={'processed': pd.eval})
        elif data_name == 'ws':
            df_ds = pd.read_csv(config['data']['processed_ws'], converters={'processed': pd.eval})
        else:
            sys.exit(1)
        print("*Building ", text_rep, "representation(s) for: ", data_name)
        extract(df_ds, tuple(map(int, min_max.split(','))))
    else:
        print("*Error: incorrect argument name.")
        sys.exit(1)
    print('*Program terminate successfully!')