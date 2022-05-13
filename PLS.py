import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, ClassifierMixin
class PLS(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.clf = PLSRegression(n_components=2)
    def fit(self, X, y):
        self.clf.fit(X,y)
        return self
    def predict(self, X):
        pr = [np.round(np.abs(item[0])) for item in self.clf.predict(X)]
        return np.array(pr)
    def predict_proba(self, X):
        p_all = []
        p_all.append([1-np.abs(item[0]) for item in self.clf.predict(X)])
        p_all.append([np.abs(item[0]) for item in self.clf.predict(X)])
        return np.transpose(np.array(p_all))
