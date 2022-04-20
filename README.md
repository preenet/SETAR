# Thai_SA_Journal
Sentimental Analysis in Thai language journal:  
version 0.1a  (4/20/2022)  
* added count and tfidf vectorizer for both unigram and bigrams.  
* added word2vec model for wisesight+thwiki 
* added word2vec model for wisesight+thwiki+kt4.0  
* added pos tagging from orchid tagging schemes
* added new features including dictionary-based with positive and negative words, along with word count  

TODO:  
- test on well-known classifers  
- concat text fit and features fit and dump as pickle (since they were peformed with differnt fitting modules) 
- apply scikit.learn.pipeline with different vectorizers on logistic regression?   


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
