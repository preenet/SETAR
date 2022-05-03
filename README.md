# Thai_SA_Journal
**State of the arts in Thai Sentimental Analysis**  
This script extracted features from three sentimental corpora: kt4.0 (data from our acquisition), thaitale, and wisesight. By training with our proposed SoTA classifiers on the kt4.0 dataset, we expect to see an improvement in the classification performance of the wisesight as well as thaitale dataset.

Several feature extraction methods were applied on text feature to both corpuses as follows:  

* Bag of words for unigram and bigrams
* TF-IDF for unigram and bigrams
* Word2Vec with TF-IDF vector (300 dimension)
* POS_tagging using 17 Orchid tags
* Dictionary-based with list of Thai positive and negative words for unigram and bigrams
  
Total of 8 text representations were exctracted for each corpus.  

Output:  
For all the feature extraction methods above, Joblib objects as sparse matrix on text feature were dumped (see /model).  

**Baseline model**
11 baseline ML models were evaluated with the KT4.0 dataset. (see /model)  
 

**Exisiting models (published)**

| Authors                                 | Method                                            | Results                             |
|-----------------------------------------|---------------------------------------------------|-------------------------------------|
| (Pasupa & Seneewong Na Ayutthaya, 2022) | Combine feature from (w2v, pose,sentic), fused DL | 0.561 F1-score                      |
| (Lowphansirikul et al., 2021)           | RoBERTa, attention spam                           | 76.19 / 67.05 micro and macro avg.  |

Dependencies
* pythainlp >= 3.06dev
* python >= 3.8.8
* gensim >= 4.1.2
* scikit-learn >= 1.0.2
* joblib => 1.1.0

# KT4.0 SA corpus  

The corpus was scraped from pantip.com's online products domain during the late 2019 to middle of 2020. The online product contains a variety of cosmetic, food, supplementary food, and skin-care products. It contains 60,081 samples following sentence tokenization with CRF on four distinct datasets.     

Preprocessed stage 1:  
* Samples that contain empty text were removed.
* Removed text that has number of word greater than 1,000 words
* Removed text that has less than 2 words  
* Removed duplicate text  
* Removed new line and tab
* Removed samples that are not a Thai language using laguage dection library  
* Sentence tokenzation was performed.  

Sentiment annotation:  
* Three Thai ligustic experts performed the annotation task. Majority vote was used to calculated the class target.  


# Wisesight corpus
For wisesight sentiment, we will use diffent split from the kaggle compitition (see source code).  
Information regarding the corpus can be found from https://github.com/PyThaiNLP/wisesight-sentiment  

# Thaitale corpus   
Information regarding the corpus can be found from https://github.com/dsmlr/40-Thai-Children-Stories  
