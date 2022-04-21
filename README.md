# Thai_SA_Journal
Sentimental Analysis in Thai language journal:  


Dependencies
* pythainlp >= 3.06dev
* python >= 3.8.8
* gensim >= 4.1.2
* scikit-learn >= 1.0.2
* joblib >= 1.1.0
* dill >= 0.31

# KT4.0 SA corpus  (UTF-8-Sig)

The corpus was scraped from pantip.com's online products domain during the late 2019 to middle of 2020. The online product contains a variety of cosmetic, food, supplementary food, and skin-care products. It contains 60,081 samples following sentence tokenization with CRF on four distinct datasets.     


Attributes:  
post_id
post_date  
user_id  
user_name 
text  
tag  
emotion
length
num_sent
sent_length  
label_1  
label_2  
label_3  
vote  

Preprocessed stage 1:  
* Samples that contain empty text were removed.
* Removed text that has number of word greater than 1,000 words
* Removed text that has less than 2 words  
* Removed duplicate text  
* Removed new line and tab
* Removed samples that are not a Thai language using laguage dection library  
* Sentence tokenzation was performed.  

Sentiment annotation:  
* Three Thai ligustic experts performed the annotation task. Next, majority vote was used to calculated the class target.  

Pre-processed stage 2 (used process_thai from pythainlp for now)
* Clean text that has white space and occurences
* Remove thai stop words and puncuations 
* Replace URL with special tag
* Replace emoticon with special tag
* lower cap for English words
* Word tokenize task performed with default pythainlp engine  


# Wisesight corpus
For wisesight sentiment, we will use diffent split from the kaggle compitition (see source code).  
Information regarding the corpus can be found from https://github.com/PyThaiNLP/wisesight-sentiment  
