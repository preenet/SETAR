"""
This script extract features for three sentimental corpora: kt4.0 (data from our acquisition), thaitale, and wisesight.
By training with our proposed SoTA classifiers and using knowledge from our kt4.0 dataset, 
we expect to see an improvement in the classification performance of the wisesight as well as thaitale dataset.

Several feature extraction methods were applied on text feature to both corpuses as follows:  

* Bag of words supports both unigram and bigrams
* TF-IDF suppots both unigram and bigrams
* Word2Vec with TF-IDF embed vectorizer (300 dimension)
* POS_tagging with three methods using 17 Orchid tags and emoji tag
* Dictionary-based with a custom list of Thai positive and negative words for the unigram and bigrams
  
Total of 8 text representations were extracted for each corpus.  
@Authors: pree.t@cmu.ac.th
"""
import sys
import pandas as pd
import numpy as np
import joblib
import src.utilities as utils

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pythainlp.tag import pos_tag_sents
from gensim.models import Word2Vec
from scipy import sparse # for converting w2v embed vector to spare matrix and targets  

from src.feature.tfidf_embedding_vectorizer import TfidfEmbeddingVectorizer
from src.feature.process_thai_text import process_text
from src.feature.pos_rule import word_tag, tag, tag_emoj
from src.visualization.visualize import top_feats_all, plot_top_feats

from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'tahoma'

# global vars
config = utils.read_config()
text_rep = ""
data_name = ""

def extract(df_kt, df_ds):

    # transfrom target string to int
    y_kt = df_kt['target'].astype('category').cat.codes
    y_ds = df_ds['target'].astype('category').cat.codes
    print("label kt:", y_kt.unique(), ", label ds", y_ds.unique())

    print("Quick peek for downstream task:")
    print(df_ds[0:10])
    print(df_ds.tail(10))
    print(df_ds.describe())

    # class distribution
    print("KT classes dist.:", df_kt.target.value_counts() / df_kt.shape[0])
    print("DS classes dist.:", df_ds.target.value_counts() / df_ds.shape[0])

    # transfrom class target 
    y_t_kt = y_kt.to_numpy().reshape(-1, 1)
    y_t_ds = y_ds.to_numpy().reshape(-1, 1)

    y_t_kt = sparse.csr_matrix(y_t_kt)
    y_t_ds = sparse.csr_matrix(y_t_ds)

    if text_rep == 'BOW':        
        bow(df_kt, y_t_kt, df_ds, y_t_ds)
    elif text_rep == 'TFIDF':    
        tfidf(df_kt, y_t_kt, df_ds, y_t_ds)
    elif text_rep == 'W2V':    
        w2v_tfidf(df_kt, y_t_kt, df_ds, y_t_ds)
    elif text_rep == 'POSBOW': 
        pos_bow(df_kt, y_t_kt, df_ds, y_t_ds)
    elif(text_rep == 'POSTFIDF'):
        pos_tfidf(df_kt, y_t_kt, df_ds, y_t_ds)
    elif(text_rep == 'DICTBOW'):
        dict_bow(df_kt, y_t_kt, df_ds, y_t_ds)
    elif(text_rep == 'DICTTFIDF'):
        dict_tfidf(df_kt, y_t_kt, df_ds, y_t_ds)
    elif(text_rep== 'ALL'):
        print("Extracted with all methods...")
        bow(df_kt, y_t_kt, df_ds, y_t_ds)
        tfidf(df_kt, y_t_kt, df_ds, y_t_ds)
        w2v_tfidf(df_kt, y_t_kt, df_ds, y_t_ds)
        pos_bow(df_kt, y_t_kt, df_ds, y_t_ds)
        pos_tfidf(df_kt, y_t_kt, df_ds, y_t_ds)
        dict_bow(df_kt, y_t_kt, df_ds, y_t_ds)
        dict_tfidf(df_kt, y_t_kt, df_ds, y_t_ds)
    else:
        sys.exit(1)
    return

def bow(df_kt, y_t_kt, df_ds, y_t_ds):
    print("Extracting BOW...")  
    # BOW with unigram and bigrams
    bow1 = CountVectorizer(tokenizer=lambda x:x.split(), ngram_range=(1, 1), min_df=5)
    bow2 = CountVectorizer(tokenizer=lambda x:x.split(), ngram_range=(2, 2), min_df=5)

    # fit kt and transform to both datasets
    bow1_fit_kt = bow1.fit(df_kt['processed'].apply(str))
    text_bow1_kt = bow1_fit_kt.transform(df_kt['processed'].apply(str))
    text_bow1_ds = bow1_fit_kt.transform(df_ds['processed'].apply(str))

    bow2_fit_kt = bow2.fit(df_kt['processed'].apply(str))
    text_bow2_kt = bow2_fit_kt.transform(df_kt['processed'].apply(str))
    text_bow2_ds = bow2_fit_kt.transform(df_ds['processed'].apply(str))

    print(text_bow1_kt.toarray().shape,  text_bow1_kt.toarray().shape)
    print(text_bow2_kt.toarray().shape,  text_bow2_kt.toarray().shape, end = " ")
    print(text_bow1_ds.toarray().shape,  text_bow1_ds.toarray().shape)
    print(text_bow2_ds.toarray().shape,  text_bow2_ds.toarray().shape, end = " ")
    
    # visualize 
    y = y_t_kt.todense()
    y = np.array(y.reshape(y.shape[0],))[0]
    plot_feats(bow1_fit_kt, text_bow1_kt, y)

    # write to disk
    write_to_disk(text_bow1_kt, y_t_kt, 'text_bow1_kt.pkl')
    write_to_disk(text_bow2_kt, y_t_kt, 'text_bow2_kt.pkl')

    write_to_disk(text_bow1_ds, y_t_ds,'text_bow1_' + data_name + '.pkl')
    write_to_disk(text_bow2_ds, y_t_ds,'text_bow2_' + data_name + '.pkl')
    return 


def tfidf(df_kt, y_t_kt, df_ds, y_t_ds):
    print("Extracting TFI-IDF...")  
    # TF-IDF with unigram and bigrams
    tfidf1 = TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=(1, 1), min_df=5)
    tfidf2 = TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=(2, 2), min_df=5)

    # fit kt and transform to both datasets
    tfidf1_fit_kt = tfidf1.fit(df_kt['processed'].apply(str))
    text_tfidf1_kt = tfidf1_fit_kt.transform(df_kt['processed'].apply(str))
    text_tfidf1_ds = tfidf1_fit_kt.transform(df_ds['processed'].apply(str))

    tfidf2_fit_kt = tfidf2.fit(df_kt['processed'].apply(str))
    text_tfidf2_kt = tfidf2_fit_kt.transform(df_kt['processed'].apply(str))
    text_tfidf2_ds = tfidf2_fit_kt.transform(df_ds['processed'].apply(str))

    print(text_tfidf1_kt.toarray().shape,  text_tfidf1_kt.toarray().shape)
    print(text_tfidf2_kt.toarray().shape,  text_tfidf2_kt.toarray().shape, end =" ")
    print(text_tfidf1_ds.toarray().shape,  text_tfidf1_ds.toarray().shape)
    print(text_tfidf2_ds.toarray().shape,  text_tfidf2_ds.toarray().shape, end =" ")

    write_to_disk(text_tfidf1_kt, y_t_kt, 'text_tfidf1_kt.pkl')
    write_to_disk(text_tfidf2_kt, y_t_kt, 'text_tfidf2_kt.pkl')

    write_to_disk(text_tfidf1_ds, y_t_ds, 'text_tfidf1_' + data_name + '.pkl')
    write_to_disk(text_tfidf2_ds, y_t_ds, 'text_tfidf2_' + data_name + '.pkl')
    return


def w2v_tfidf(df_kt, y_t_kt, df_ds, y_t_ds):
    print("Extracting W2V-TFIDF...")  
    # create word2vec for kt corpus
    w2v_kt = Word2Vec(vector_size=300, min_count=1, window=4, workers=4)
    w2v_kt.build_vocab(df_kt['processed'])
    w2v_kt.train(df_kt['processed'], total_examples=w2v_kt.corpus_count, epochs=100)
    print(w2v_kt.wv.most_similar("บะหมี่"))

    w2v_tfidf_emb_kt = TfidfEmbeddingVectorizer(w2v_kt)
    w2v_tifdf_fit_kt = w2v_tfidf_emb_kt.fit(df_kt['processed'])

    # transfrom on both corpuses
    text_w2v_tfidf_kt = w2v_tifdf_fit_kt.transform(df_kt['processed'])
    text_w2v_tfidf_ds = w2v_tifdf_fit_kt.transform(df_ds['processed'])

    write_to_disk(sparse.csr_matrix(text_w2v_tfidf_kt), y_t_kt, 'text_w2v_tfidf_kt.pkl')
    write_to_disk(sparse.csr_matrix(text_w2v_tfidf_ds), y_t_ds, 'text_w2v_tfidf_' + data_name + '.pkl')
    return 

def pos_bow(df_kt, y_t_kt, df_ds, y_t_ds):
    print("Extracting POS_BOW...")   
    pos = pos_tag_sents(df_kt['processed'].tolist(), corpus='orchid_ud')
    pos = tag_emoj(pos)
    df_kt['pos_tag1'] = pd.DataFrame(tag(pos))
    df_kt['pos_tag2'] = pd.DataFrame(word_tag(pos))

    pos = pos_tag_sents(df_ds['processed'].tolist(), corpus='orchid_ud')
    pos = tag_emoj(pos)
    df_ds['pos_tag1'] = pd.DataFrame(tag(pos))
    df_ds['pos_tag2'] = pd.DataFrame(word_tag(pos))

    print(df_ds[['processed', 'pos_tag1']].iloc[1000:1010])
    print(df_ds['pos_tag2'].iloc[1000:1010])

    # create bow vectors
    bow1 = CountVectorizer(ngram_range=(1, 1))
    #bow2 = CountVectorizer(ngram_range=(2, 2))

    text_pos_bow1_fit_kt = bow1.fit(df_kt['pos_tag1'])
    text_pos_bow1_kt = text_pos_bow1_fit_kt.transform(df_kt['pos_tag1'])
    text_pos_bow1_ds = text_pos_bow1_fit_kt.transform(df_ds['pos_tag1'])

    # text_pos_bow2_fit_kt = bow2.fit(df_kt['pos_tag1'])
    # text_pos_bow2_kt = text_pos_bow2_fit_kt.transform(df_kt['pos_tag2'])
    # text_pos_bow2_ds = text_pos_bow2_fit_kt.transform(df_ds['pos_

    print(text_pos_bow1_kt.toarray().shape,  text_pos_bow1_kt.toarray().shape)
    print(text_pos_bow1_ds.toarray().shape,  text_pos_bow1_ds.toarray().shape)

    # print(text_pos_bow2_kt.toarray().shape,  text_pos_bow2_kt.toarray().shape)
    # print(text_pos_bow2_ds.toarray().shape,  text_pos_bow2_ds.toarray().shape)

    write_to_disk(text_pos_bow1_kt, y_t_kt, 'text_pos_bow1_kt.pkl')
    write_to_disk(text_pos_bow1_ds, y_t_ds, 'text_pos_bow1_' + data_name + '.pkl')

    # write_to_disk(text_pos_bow2_kt, y_t_kt, 'text_pos_bow2_kt.pkl')
    # write_to_disk(text_pos_bow2_ds, y_t_ds, 'text_pos_bow2_' + data_name + '.pkl')
    return

def pos_tfidf(df_kt, y_t_kt, df_ds, y_t_ds):
    print("Extracting POS_TF-IDF...")  
    # create tfidf vectors
    tfidf1 = TfidfVectorizer(ngram_range=(1, 1))
    tfidf2 = TfidfVectorizer(ngram_range=(2, 2))

    text_pos_tfidf1_fit_kt = tfidf1.fit(df_kt['pos_tag1'])
    text_pos_tfidf1_kt = text_pos_tfidf1_fit_kt.transform(df_kt['pos_tag1'])
    text_pos_tfidf1_ds = text_pos_tfidf1_fit_kt.transform(df_ds['pos_tag1'])

    text_pos_tfidf2_fit_kt = tfidf2.fit(df_kt['pos_tag1'])
    text_pos_tfidf2_kt = text_pos_tfidf2_fit_kt.transform(df_kt['pos_tag2'])
    text_pos_tfidf2_ds = text_pos_tfidf2_fit_kt.transform(df_ds['pos_tag2'])

    print(text_pos_tfidf1_kt.toarray().shape,  text_pos_tfidf1_kt.toarray().shape)
    print(text_pos_tfidf1_ds.toarray().shape,  text_pos_tfidf1_ds.toarray().shape)

    print(text_pos_tfidf2_kt.toarray().shape,  text_pos_tfidf2_kt.toarray().shape)
    print(text_pos_tfidf2_ds.toarray().shape,  text_pos_tfidf2_ds.toarray().shape)

    write_to_disk(text_pos_tfidf1_kt, y_t_kt, 'text_pos_tfidf1_kt.pkl')
    write_to_disk(text_pos_tfidf2_kt, y_t_kt, 'text_pos_tfidf2_kt.pkl')

    write_to_disk(text_pos_tfidf1_ds, y_t_ds, 'text_pos_tfidf1_' + data_name + '.pkl')
    write_to_disk(text_pos_tfidf2_ds, y_t_ds, 'text_pos_tfidf2_' + data_name + '.pkl')
    return

def dict_bow(df_kt, y_t_kt, df_ds, y_t_ds):
    print("Extracting DICT_BOW...")  
    my_vocabs = get_dict_vocab()
    bow1 = CountVectorizer(tokenizer=lambda x:x.split(), ngram_range=(1, 1))
    bow2 = CountVectorizer(tokenizer=lambda x:x.split(), ngram_range=(2, 2))

    text_dict_bow1_fit = bow1.fit(my_vocabs)
    text_dict_bow1_kt = text_dict_bow1_fit.transform(df_kt['processed'].apply(str))
    text_dict_bow1_ds = text_dict_bow1_fit.transform(df_ds['processed'].apply(str))

    text_dict_bow2_fit = bow2.fit(my_vocabs)
    text_dict_bow2_kt = text_dict_bow2_fit.transform(df_kt['processed'].apply(str))
    text_dict_bow2_ds = text_dict_bow2_fit.transform(df_ds['processed'].apply(str))

    print(text_dict_bow1_kt.toarray().shape,  text_dict_bow1_kt.toarray().shape)
    print(text_dict_bow1_ds.toarray().shape,  text_dict_bow1_ds.toarray().shape)

    print(text_dict_bow2_kt.toarray().shape,  text_dict_bow2_kt.toarray().shape)
    print(text_dict_bow2_ds.toarray().shape,  text_dict_bow2_ds.toarray().shape)

    write_to_disk(text_dict_bow1_kt, y_t_kt, 'text_dict_bow1_kt.pkl')
    write_to_disk(text_dict_bow2_kt, y_t_kt, 'text_dict_bow2_kt.pkl')

    write_to_disk(text_dict_bow1_ds, y_t_ds, 'text_dict_bow1_' + data_name + '.pkl')
    write_to_disk(text_dict_bow2_ds, y_t_ds, 'text_dict_bow2_' + data_name + '.pkl')
    return

def dict_tfidf(df_kt, y_t_kt, df_ds, y_t_ds):
    print("Extracting DICT_TF-IDF...")  
    my_vocabs = get_dict_vocab()
    tfidf1 = TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=(1, 1))
    tfidf2 = TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=(2, 2))

    text_dict_tfidf1_fit = tfidf1.fit(my_vocabs)
    text_dict_tfidf1_kt = text_dict_tfidf1_fit.transform(df_kt['processed'].apply(str))
    text_dict_tfidf1_ds = text_dict_tfidf1_fit.transform(df_ds['processed'].apply(str))

    text_dict_tfidf2_fit = tfidf2.fit(my_vocabs)
    text_dict_tfidf2_kt = text_dict_tfidf2_fit.transform(df_kt['processed'].apply(str))
    text_dict_tfidf2_ds = text_dict_tfidf2_fit.transform(df_ds['processed'].apply(str))

    print(text_dict_tfidf1_kt.toarray().shape,  text_dict_tfidf1_kt.toarray().shape)
    print(text_dict_tfidf1_ds.toarray().shape,  text_dict_tfidf1_ds.toarray().shape)

    print(text_dict_tfidf2_kt.toarray().shape,  text_dict_tfidf2_kt.toarray().shape)
    print(text_dict_tfidf2_ds.toarray().shape,  text_dict_tfidf2_ds.toarray().shape)

    write_to_disk(text_dict_tfidf1_kt, y_t_kt, 'text_dict_tfidf1_kt.pkl')
    write_to_disk(text_dict_tfidf2_kt, y_t_kt, 'text_dict_tfidf2_kt.pkl')

    write_to_disk(text_dict_tfidf1_ds, y_t_ds, 'text_dict_tfidf1_' + data_name + '.pkl')
    write_to_disk(text_dict_tfidf2_ds, y_t_ds, 'text_dict_tfidf2_' + data_name + '.pkl')
    return

def get_dict_vocab():
    # load list of our custom positive and negative words
    data_path_dict = config['data']['raw_dict']
    try:
        with open(data_path_dict + 'pos_words.txt', encoding='utf-8') as f:
            pos_words = [line.rstrip('\n') for line in f]
    except IOError as e:
        print("Error: can't open file to read", e)
        sys.exit(1)
    f.close()

    try: 
        with open(data_path_dict + 'neg_words.txt', encoding='utf-8') as f:
            neg_words = [line.rstrip('\n') for line in f]
            pos_words = list(set(pos_words))
            neg_words = list(set(neg_words))
            my_vocabs = pos_words + neg_words
            print('dict size: ', len(my_vocabs))
    except IOError as e:
        print("Error: can't open file to read", e)
        sys.exit(1)
    f.close()
    return my_vocabs

def write_to_disk(text_rep, y, file_name):
    joblib.dump(np.hstack((text_rep, y)), config['output_scratch'] + file_name)
    return

def plot_feats(vectorizer, X, y):
    features = vectorizer.get_feature_names()
    tf = top_feats_all(X.toarray(), y, features)
    plot_top_feats(tf)
    return

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("*Error: incorrect number of arguments.")
        print("*Usage: ", config['feature']['build_method'] , "| [dataset name]")
        sys.exit(1)

    elif sys.argv[1] in config['feature']['build_method'] and sys.argv[2] in config['data']['name']: 
        text_rep = sys.argv[1]
        data_name = sys.argv[2]

        df_kt = pd.read_csv(config['data']['processed_kt'], converters={'processed': pd.eval})
        if data_name == 'ws':
            print("Loading and converting from csv...")
            df_ds = pd.read_csv(config['data']['processed_ws'], converters={'processed': pd.eval})
        elif data_name == 'tt':
            print("Loading and converting from csv...")
            df_ds = pd.read_csv(config['data']['processed_tt'], converters={'processed': pd.eval})
    
        print("*Building ", text_rep, "representation(s) for: ", data_name)
        extract(df_kt, df_ds)
    else:
        print("*Error: incorrect argument name.")
        sys.exit(1)
    print('*Program terminate successfully!')