"""
## Feature selection using two-stage filtering method  
We applied a two-stage filtering feature selection method for both bow and tfi-df text representations only.    
filter 1 - Variance Threshold Feature Selection
filter 2 - Univariate Selection using SelectKBest

pree.t@cmu.ac.th  
"""
from distutils.command.config import config
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
import matplotlib.pyplot as plt
import src.utilities as utils

class SelectFeatures:
    
    def __init__(self, vi_dim, mi_dim):
        self.vi_dim = vi_dim
        self.mi_dim = mi_dim

    def twosteps_fs(self, X, y):
            vt = VarianceThreshold()
            vt.fit(X)
            vi_scores = vt.variances_

            idx = np.flip(np.argsort(vi_scores))
            tmp = np.take(X, idx.flatten(), axis=1)        
            X_vt = tmp[:, :self.vi_dim]

            mi_scores = mutual_info_classif(X_vt, np.ravel(y), random_state=0)
            mi_idx = np.flip(np.argsort(vi_scores))
            tmp = np.take(X_vt, mi_idx.flatten(), axis=1)        
            X_vt_mi = tmp[:, :self.mi_dim]
            return X_vt, vi_scores, X_vt_mi, mi_scores

    def plot_feature_scores(self, scores, threshold):
        plt.figure(figsize=(9, 7), dpi=80) 
        ax = plt.axes()
        ax.axhline(threshold, ls='dotted', c='r')
        ax.plot(scores)
        return
        

# if __name__ == "__main__":
#     config = utils.read_config()
#     sf = SelectFeatures( config['feature']['selection']['vi_dim'], config['feature']['selection']['vi_dim'] )
#     print(sf.mi_dim)

