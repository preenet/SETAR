# Thai_SA_Journal Dev.
=======================  
**State of the arts in Thai Sentimental Analysis**  

Several feature extraction methods were applied on text feature to all the corpuses as follows:  

    * Bag of words for unigram and bigrams
    * TF-IDF for unigram and bigrams
    * Word2Vec with TF-IDF vector (300 dimension)
    * POS_tagging using 17 Orchid tags
    * Dictionary-based with list of Thai positive and negative words for unigram and bigrams
    
Total of 8 text representations were extracted for each corpus.  

Output:  
For all the feature extraction methods above, Joblib objects as sparse matrix on text feature were dumped (see data folders).  

#Todo
งาน อ.เปิ้ล  
1. Run Wongnai  
2. ปั้น  Wanchan Stack   (WS /  TT /  Wongnai)
3. ปั้น Wanchan Boosting Stack Ensemble (God Mode)  (WS/ TT/ Wongnai)      เย๊อะ แฮะ ฮ่าๆๆ
งาน อ.ปรี
1. POSTAG (ทุก Data)
2. KT Baseline
3. Deep Baseline (ทุก Data)


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


