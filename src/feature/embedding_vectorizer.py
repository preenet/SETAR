from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingVectorizer(object):
    def __init__(self, model, type, min_max):
        self.word2vec = {w: vec for w, vec in zip(model.wv.index_to_key, model.wv.vectors)}
        self.word2weight = None
        self.dim = model.vector_size
        self.type = type
        self.min_max = min_max

    def fit(self, X):
        if self.type == "tfidf":
            tfidf = TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=self.min_max, min_df=1, sublinear_tf=True)
            tfidf.fit(X)
            max_idf = max(tfidf.idf_)
            self.word2weight = defaultdict(lambda: max_idf,[(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        if self.type == "tfidf":
            return sparse.csr_matrix([np.mean([self.word2vec[w] * self.word2weight[w] for w in words.split() if w in self.word2vec] 
                                              or [np.zeros(self.dim)], axis=0) for words in X], dtype="float32")
        else:
            return sparse.csr_matrix([np.mean([self.word2vec[w] for w in words.split() if w in self.word2vec] 
                                              or [np.zeros(self.dim)], axis=0) for words in X], dtype="float32")
