# SETAR  
=======================  
**State of the arts in Thai Sentimental Analysis**  

Several feature extraction methods were applied on text feature to all the corpuses as follows:  

    * Bag of words for 1 gram, 2 grams, and 1 and 2 grams.
    * TF-IDF for 1 gram, 2 grams, and 1 and 2 grams.
    * Word2Vec with average and TF-IDF vectors (300 dimension).  
    * 5 schemes of POS_tagging using 17 Orchid tags.  
    * Dictionary-based with list of Thai positive and negative words for unigram.  
    
Total of 14 text representations were extracted for each corpus.  


**Existing models (published)**

| Authors                                 | Method                                            | Results                             |
|-----------------------------------------|---------------------------------------------------|-------------------------------------|
| (Pasupa & Seneewong Na Ayutthaya, 2022) | Combine feature from (w2v, pose, sentic), fused DL | 0.561 F1-score                      |
| (Lowphansirikul et al., 2021)           | RoBERTa, attention spam                           | 76.19 / 67.05 micro and macro avg.  |



# KT4.0 SA corpus  

The corpus was scraped from pantip.com's online products domain during the late 2019 to middle of 2020. The online product contains a variety of cosmetic, food, supplementary food, and skin-care products. It contains 60,081 samples following sentence tokenization with CRF on four distinct datasets.     

Preprocessed stage 1:  
    * Samples that contain empty text were removed.
    * Removed text that has number of word greater than 1,000 words
    * Removed text that has less than 2 words  
    * Removed duplicate text  
    * Removed new line and tab
    * Removed samples that are not a Thai language using language detection library  
    * Sentence tokenization was performed.  

Sentiment annotation:  
    * Three Thai linguistic experts performed the annotation task. Majority vote was used to calculated the class target.  


# Wisesight corpus
For wisesight sentiment, we will use different split from the kaggle competition (see source code).  
Information regarding the corpus can be found from https://github.com/PyThaiNLP/wisesight-sentiment  

# Thaitale corpus   
Information regarding the corpus can be found from https://github.com/dsmlr/40-Thai-Children-Stories  

# Thai Toxic tweet corpus  
Information regarding the corpus can be found from https://huggingface.co/datasets/thai_toxicity_tweet

# How to run Deep learning:    
# CNN-TX (Done)
# CNN-TT (Tuning)  

```


```