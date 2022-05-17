import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

class TfidfEmbeddingVectorizer(object):
    def __init__(self, model, min_max):
        w2v = {w: vec for w, vec in zip(model.wv.index_to_key, model.wv.vectors)}
        self.word2vec = w2v
        self.word2weight = None
        self.dim = model.vector_size
        self.min_max = min_max
    
    def fit(self, X):
        word_list = []
        
        for word in X:
             word_list.append(" ".join(word))

        tfidf = TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=self.min_max, min_df=20, sublinear_tf=True)
        tfidf.fit(word_list)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return sparse.csr_matrix([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])