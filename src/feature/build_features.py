"""
This script extract features from three sentimental corpora: kt4.0 (data from our acquisition), thaitale, and wisesight. 
By training with our proposed SoTA classifiers on the kt4.0 dataset, 
we expect to see an improvement in the classification performance of the wisesight as well as thaitale dataset.

Several feature extraction methods were applied on text feature to both corpuses as follows:  

* Bag of words for unigram and bigrams
* TF-IDF for unigram and bigrams
* Word2Vec with TF-IDF vector (300 dimension)
* POS_tagging using 17 Orchid tags
* Dictionary-based with list of Thai positive and negative words for unigram and bigrams
  
Total of 8 text representations were exctracted for each corpus.  
@Authors: pree.t@cmu.ac.th
"""

import os, sys
import pandas as pd
import numpy as np
import src.utilities as utils
from matplotlib import pyplot as plt
from src.feature.process_thai_text import process_text
plt.rcParams['font.family'] = 'tahoma'


# get config file
config = utils.read_config()

# output folder to be put in
out_path = os.path.join(config['output'])


def extract(data_name):
    print("Extracting:", data_name)
    data_path_dict = os.path.join(config['data']['raw_dict'])

    print("Pre-processing text...")
    if data_name == 'kt':
        data_path = os.path.join(config['data']['raw_kt'])
        df = pd.read_csv(data_path)
        df = df.rename(columns={'text': 'texts'})
    elif data_name == 'ws':
        data_path = os.path.join(config['data']['raw_ws'])
        df = pd.read_csv(data_path)
    else:
        sys.exit("No such data name.")
    
    df['processed'] = df['texts'].apply(str).apply(process_text)
    print(df[0:10])
    print(df.head(10))
    print(df.describe())
    return df
    
def bow1(df):
    return

def bow2(df):
    return

def tfidf1(df):
    return

def tfidf2(df):
    return

def w2v(df):
    return

def pos_bow1():
    return

def dict_bow1():
    return

def dict_bow2():
    return

def dict_tfidf1():
    return

def dict_tfidf2():
    return
        

if __name__ == "__main__":
    print("Building features...")
    df = extract('kt')
    
    print("Finished building features!")