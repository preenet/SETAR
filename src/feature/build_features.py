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
import joblib
import src.utilities as utils

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pythainlp.tag import pos_tag_sents
from gensim.models import Word2Vec
from scipy import sparse 

from src.feature.tfidf_embedding_vectorizer import TfidfEmbeddingVectorizer
from src.feature.pos_rule import word_tag, tag, tag_emoj
from src.visualization.visualize import top_feats_all, plot_top_feats

from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'tahoma'

# global vars
config = utils.read_config()
text_rep = ""
data_name = ""

def extract(df_ds):

    # transfrom target string to int
    y = df_ds['target'].astype('category').cat.codes
    print("label: ", y.unique())

    print("Quick peek for downstream task:")
    print(df_ds[0:10])
    print(df_ds.tail(10))
    print(df_ds.describe())

    # class distribution
    print("DS classes dist.:", df_ds.target.value_counts() / df_ds.shape[0])

    # transfrom class target 
    yt = y.to_numpy().reshape(-1, 1)
    yt = sparse.csr_matrix(yt)

    if text_rep == 'BOW':        
        bow(df_ds, yt)
    elif text_rep == 'TFIDF':    
        tfidf(df_ds, yt)
    elif text_rep == 'W2V':    
        w2v_tfidf(df_ds, yt)
    elif text_rep == 'POSBOW': 
        pos_bow(df_ds, yt)
    elif(text_rep == 'POSTFIDF'):
        pos_tfidf(df_ds, yt)
    elif(text_rep == 'DICTBOW'):
        dict_bow(df_ds, yt)
    elif(text_rep == 'DICTTFIDF'):
        dict_tfidf(df_ds, yt)
    elif(text_rep== 'ALL'):
        print("Extracted with all methods...")
        bow(df_ds, yt)
        tfidf(df_ds, yt)
        w2v_tfidf(df_ds, yt)
        pos_bow(df_ds, yt)
        pos_tfidf(df_ds, yt)
        dict_bow(df_ds, yt)
        dict_tfidf(df_ds, yt)
    else:
        sys.exit(1)
    return

def bow(df_ds, yt):
    print("Extracting BOW...")  
    # BOW with unigram and bigrams
    bow1 = CountVectorizer(tokenizer=lambda x:x.split(), ngram_range=(1, 1), min_df=5)
    bow2 = CountVectorizer(tokenizer=lambda x:x.split(), ngram_range=(2, 2), min_df=5)

    # fit kt and transform to both datasets
    bow1_fit = bow1.fit(df_ds['processed'].apply(str))
    text_bow1 = bow1_fit.transform(df_ds['processed'].apply(str))

    bow2_fit = bow2.fit(df_ds['processed'].apply(str))
    text_bow2 = bow2_fit.transform(df_ds['processed'].apply(str))
    
    print(text_bow1.toarray().shape)
    print(text_bow2.toarray().shape)
    
    # visualize 
    # y = yt.todense()
    # y = np.array(y.reshape(y.shape[0],))[0]
    # plot_feats(bow1_fit, text_bow1, y)

    # write to disk
    write_to_disk(text_bow1, yt,'text_bow1_' + data_name + '.pkl')
    write_to_disk(text_bow2, yt,'text_bow2_' + data_name + '.pkl')
    return 

def tfidf(df_ds, yt):
    print("Extracting TFI-IDF...")  
    # TF-IDF with unigram and bigrams
    tfidf1 = TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=(1, 1), min_df=5)
    tfidf2 = TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=(2, 2), min_df=5)

    # fit kt and transform to both datasets
    tfidf1_fit = tfidf1.fit(df_ds['processed'].apply(str))
    text_tfidf1 = tfidf1_fit.transform(df_ds['processed'].apply(str))

    tfidf2_fit = tfidf2.fit(df_ds['processed'].apply(str))
    text_tfidf2 = tfidf2_fit.transform(df_ds['processed'].apply(str))

    print(text_tfidf1.toarray().shape)
    print(text_tfidf2.toarray().shape)

    write_to_disk(text_tfidf1, yt, 'text_tfidf1_' + data_name + '.pkl')
    write_to_disk(text_tfidf2, yt, 'text_tfidf2_' + data_name + '.pkl')
    return

def w2v_tfidf(df_ds, yt):
    print("Extracting W2V-TFIDF...")  
    # create word2vec for kt corpus
    w2v = Word2Vec(vector_size=300, min_count=1, window=4, workers=4)
    w2v.build_vocab(df_ds['processed'])
    w2v.train(df_ds['processed'], total_examples=w2v.corpus_count, epochs=100)

    w2v_tfidf_emb = TfidfEmbeddingVectorizer(w2v)
    w2v_tifdf_fit = w2v_tfidf_emb.fit(df_ds['processed'])
    text_w2v_tfidf = w2v_tifdf_fit.transform(df_ds['processed'])
    
    write_to_disk(sparse.csr_matrix(text_w2v_tfidf), yt, 'text_w2v_tfidf_' + data_name + '.pkl')
    return 

def pos_bow(df_ds, yt):
    print("Extracting POS_BOW...")   

    pos = pos_tag_sents(df_ds['processed'].tolist(), corpus='orchid_ud')
    pos = tag_emoj(pos)
    df_ds['pos_tag1'] = pd.DataFrame(tag(pos))
    df_ds['pos_tag2'] = pd.DataFrame(word_tag(pos))

    print(df_ds[['processed', 'pos_tag1']].iloc[1000:1010])
    print(df_ds['pos_tag2'].iloc[1000:1010])

    # create bow vectors
    bow1 = CountVectorizer(ngram_range=(1, 1))
    #bow2 = CountVectorizer(ngram_range=(2, 2))

    text_pos_bow1_fit = bow1.fit(df_ds['pos_tag1'])
    text_pos_bow1 = text_pos_bow1_fit.transform(df_ds['pos_tag1'])

    # text_pos_bow2_fit = bow2.fit(df_ds['pos_tag1'])
    # text_pos_bow2 = text_pos_bow2_fit.transform(df_ds['pos_tag1'])

    print(text_pos_bow1.toarray().shape)

    # print(text_pos_bow2.toarray().shape,  text_pos_bow2_ds.toarray().shape)

    write_to_disk(text_pos_bow1, yt, 'text_pos_bow1_' + data_name + '.pkl')

    # write_to_disk(text_pos_bow2, yt, 'text_pos_bow2_' + data_name + '.pkl')
    return

def pos_tfidf(df_ds, yt):
    print("Extracting POS_TF-IDF...")  
    # create tfidf vectors
    tfidf1 = TfidfVectorizer(ngram_range=(1, 1))
    tfidf2 = TfidfVectorizer(ngram_range=(2, 2))

    text_pos_tfidf1_fit = tfidf1.fit(df_ds['pos_tag1'])
    text_pos_tfidf1 = text_pos_tfidf1_fit.transform(df_ds['pos_tag1'])

    text_pos_tfidf2_fit = tfidf2.fit(df_ds['pos_tag1'])
    text_pos_tfidf2 = text_pos_tfidf2_fit.transform(df_ds['pos_tag2'])

    print(text_pos_tfidf1.toarray().shape)
    print(text_pos_tfidf2.toarray().shape)


    write_to_disk(text_pos_tfidf1, yt, 'text_pos_tfidf1_' + data_name + '.pkl')
    write_to_disk(text_pos_tfidf2, yt, 'text_pos_tfidf2_' + data_name + '.pkl')
    return

def dict_bow(df_ds, yt):
    print("Extracting DICT_BOW...")  
    my_vocabs = get_dict_vocab()
    bow1 = CountVectorizer(vocabulary=my_vocabs, tokenizer=lambda x:x.split(), ngram_range=(1, 1))
    bow2 = CountVectorizer(vocabulary=my_vocabs, tokenizer=lambda x:x.split(), ngram_range=(2, 2))

    text_dict_bow1_fit = bow1.fit(df_ds['processed'].apply(str))
    text_dict_bow1 = text_dict_bow1_fit.transform(df_ds['processed'].apply(str))

    text_dict_bow2_fit = bow2.fit(df_ds['processed'].apply(str))
    text_dict_bow2 = text_dict_bow2_fit.transform(df_ds['processed'].apply(str))

    print(text_dict_bow1.toarray().shape)
    print(text_dict_bow2.toarray().shape)


    write_to_disk(text_dict_bow1, yt, 'text_dict_bow1_' + data_name + '.pkl')
    write_to_disk(text_dict_bow2, yt, 'text_dict_bow2_' + data_name + '.pkl')
    return

def dict_tfidf(df_ds, yt):
    print("Extracting DICT_TF-IDF...")  
    my_vocabs = get_dict_vocab()
    tfidf1 = TfidfVectorizer(vocabulary=my_vocabs, tokenizer=lambda x:x.split(), ngram_range=(1, 1))
    tfidf2 = TfidfVectorizer(vocabulary=my_vocabs, tokenizer=lambda x:x.split(), ngram_range=(2, 2))

    text_dict_tfidf1_fit = tfidf1.fit(df_ds['processed'].apply(str))
    text_dict_tfidf1 = text_dict_tfidf1_fit.transform(df_ds['processed'].apply(str))

    text_dict_tfidf2_fit = tfidf2.fit(df_ds['processed'].apply(str))
    text_dict_tfidf2 = text_dict_tfidf2_fit.transform(df_ds['processed'].apply(str))

    print(text_dict_tfidf1.toarray().shape)
    print(text_dict_tfidf2.toarray().shape)

    write_to_disk(text_dict_tfidf1, yt, 'text_dict_tfidf1_' + data_name + '.pkl')
    write_to_disk(text_dict_tfidf2, yt, 'text_dict_tfidf2_' + data_name + '.pkl')
    return

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

        print("Loading and converting from csv...")
        df_ds = pd.read_csv(config['data']['processed_kt'], converters={'processed': pd.eval})    
        print("*Building ", text_rep, "representation(s) for: ", data_name)
        extract(df_ds)
    else:
        print("*Error: incorrect argument name.")
        sys.exit(1)
    print('*Program terminate successfully!')