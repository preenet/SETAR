#df_ds = pd.read_csv('/kaggle/input/processed-wisesight/wisesight.txt')
#y_ds = df_ds['target'].astype('category').cat.codes
#Xo = [' '.join(process_text(item))  for item in df_ds['text']]
#yo = y_ds.to_numpy()
#import joblib
#joblib.dump((Xo, yo), "ws.sav")
from pathlib import Path

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import src.utilities as utils
from scipy import sparse

configs = utils.read_config()
root = utils.get_project_root()

data_path = str(Path.joinpath(root, configs['data']['wangcha_kt']))
# Testing Only 
#idx1 = list(np.random.choice(np.where(yo==0)[0], 20, replace=False))
#idx2 = list(np.random.choice(np.where(yo==1)[0], 20, replace=False))
#idx3 = list(np.random.choice(np.where(yo==2)[0], 20, replace=False))
#idx4 = list(np.random.choice(np.where(yo==3)[0], 20, replace=False))
#idx = np.array(idx1+idx2+idx3+idx4)
#Xo = np.array(Xo)
#Xo = Xo[idx]
#yo = yo[idx]
SEED = [item for item in range(0,10)]
file = open('PLS.py','w')
file.write('import numpy as np'+"\n")
file.write('from sklearn.cross_decomposition import PLSRegression'+"\n")
file.write('from sklearn.base import BaseEstimator, ClassifierMixin'+"\n")
file.write('class PLS(BaseEstimator, ClassifierMixin):'+"\n")
file.write('    def __init__(self):'+"\n")
file.write('        self.clf = PLSRegression(n_components=2)'+"\n")
file.write('    def fit(self, X, y):'+"\n")
file.write('        self.clf.fit(X,y)'+"\n")
file.write('        return self'+"\n")
file.write('    def predict(self, X):'+"\n")
file.write('        pr = [min(max(np.round(item[0]),0.000001),0.999999) for item in self.clf.predict(X)]'+"\n")
file.write('        return np.array(pr)'+"\n")
file.write('    def predict_proba(self, X):'+"\n")
file.write('        p_all = []'+"\n")
file.write('        ptmp = np.array([min(max(item[0],0.000001),0.999999) for item in self.clf.predict(X)],dtype=float)'+"\n")
file.write('        p_all.append(1-ptmp)'+"\n")
file.write('        p_all.append(ptmp)'+"\n")
file.write('        return np.transpose(np.array(p_all))'+"\n")
file.close()

import gensim
#from sklearn.svm import SVC
from daal4py.sklearn.svm import SVC
from lightgbm import LGBMClassifier
from PLS import PLS
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef  # average == 'macro'.
from sklearn.metrics import \
    roc_auc_score  # multiclas 'ovo' average == 'macro'.
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def test(clf, X, y, Xt, yt):
    train_X, test_X = X, Xt
    train_y, test_y = y, yt
    clf.fit(train_X, train_y)        
    p = clf.predict(test_X)
    pr = clf.predict_proba(test_X)
    ACC = accuracy_score(test_y,p)
    SENS = precision_score(test_y,p, average='macro')
    SPEC = recall_score(test_y,p, average='macro')
    MCC = matthews_corrcoef(test_y,p)
    AUC = roc_auc_score(test_y,pr,multi_class='ovo',average='macro')
    F1 = 2*SENS*SPEC/(SENS+SPEC)
    return ACC, SENS, SPEC, MCC, AUC, F1

from sklearn.datasets import load_svmlight_file


def get_data(idx):
    data = load_svmlight_file(data_path + "\\" + "traindata_"+str(idx)+".scl", zero_based=False)
    data1 = load_svmlight_file(data_path + "\\" + "testdata_"+str(idx)+".scl", zero_based=False)
    return data[0].toarray(), data[1], data1[0].toarray(), data1[1]

#SEED = int(sys.argv[-2])
for item in SEED:
    iname = 'WANCHAN'

    Xa, ya, Xt, yt = get_data(item)
    idx = int(round(len(ya)*0.75))
    X_train_val = Xa[0:idx, :]
    y = ya[0:idx]
    X_val_val = Xa[idx:, :]
    yv = ya[idx:]

    scaler = MaxAbsScaler()
    scaler.fit(X_train_val)
    X = sparse.csr_matrix(np.vstack((scaler.transform(X_train_val),scaler.transform(X_val_val))))
    y = np.hstack((y,yv))
    Xt = sparse.csr_matrix(scaler.transform(Xt))

    featx = []
    ix = []
    nr_fold = 5
    for i in range(0, len(y)):
        ix.append(i)
    ix = np.array(ix)

    for i in range(1):
        Xs = X
        feat = np.zeros((X.shape[0],3),dtype=float)
        for j in range(0, nr_fold):
            train_ix = ((ix % nr_fold) != j)
            test_ix = ((ix % nr_fold) == j)
            train_X, test_X = Xs[train_ix], Xs[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            clf = SVC(random_state=0, probability=True)
            clf.fit(train_X, train_y)
            pr = clf.predict_proba(test_X)
            feat[test_ix] = pr
        if (i == 0):
            featx = feat
        else:
            featx = np.concatenate((featx,feat),axis=1)

        feat = np.zeros((X.shape[0],3),dtype=float)
        for j in range(0, nr_fold):
            train_ix = ((ix % nr_fold) != j)
            test_ix = ((ix % nr_fold) == j)
            train_X, test_X = Xs[train_ix], Xs[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            clf = RandomForestClassifier(random_state=0)
            clf.fit(train_X, train_y)
            pr = clf.predict_proba(test_X)
            feat[test_ix] = pr
        featx = np.concatenate((featx,feat),axis=1)

        feat = np.zeros((X.shape[0],3),dtype=float)
        for j in range(0, nr_fold):
            train_ix = ((ix % nr_fold) != j)
            test_ix = ((ix % nr_fold) == j)
            train_X, test_X = Xs[train_ix], Xs[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            clf = ExtraTreesClassifier(random_state=0)
            clf.fit(train_X, train_y)
            pr = clf.predict_proba(test_X)
            feat[test_ix] = pr
        featx = np.concatenate((featx,feat),axis=1)

        feat = np.zeros((X.shape[0],3),dtype=float)
        for j in range(0, nr_fold):
            train_ix = ((ix % nr_fold) != j)
            test_ix = ((ix % nr_fold) == j)
            train_X, test_X = Xs[train_ix], Xs[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            clf = LGBMClassifier(random_state=0) 
            clf.fit(train_X, train_y)
            pr = clf.predict_proba(test_X)
            feat[test_ix] = pr
        featx = np.concatenate((featx,feat),axis=1)

        feat = np.zeros((X.shape[0],3),dtype=float)
        for j in range(0, nr_fold):
            train_ix = ((ix % nr_fold) != j)
            test_ix = ((ix % nr_fold) == j)
            train_X, test_X = Xs[train_ix], Xs[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            clf = MLPClassifier(random_state=0)
            clf.fit(train_X, train_y)
            pr = clf.predict_proba(test_X)
            feat[test_ix] = pr
        featx = np.concatenate((featx,feat),axis=1)

        feat = np.zeros((X.shape[0],3),dtype=float)
        for j in range(0, nr_fold):
            train_ix = ((ix % nr_fold) != j)
            test_ix = ((ix % nr_fold) == j)
            train_X, test_X = Xs[train_ix], Xs[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            clf = LogisticRegression(random_state=0)
            clf.fit(train_X, train_y)
            pr = clf.predict_proba(test_X)
            feat[test_ix] = pr
        featx = np.concatenate((featx,feat),axis=1)

        feat = np.zeros((X.shape[0],3),dtype=float)
        for j in range(0, nr_fold):
            train_ix = ((ix % nr_fold) != j)
            test_ix = ((ix % nr_fold) == j)
            train_X, test_X = Xs[train_ix], Xs[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            clf = OneVsRestClassifier(PLS())
            clf.fit(train_X.toarray(), train_y)
            pr = clf.predict_proba(test_X.toarray())
            feat[test_ix] = pr
        featx = np.concatenate((featx,feat),axis=1)

    from sklearn.datasets import dump_svmlight_file
    dump_svmlight_file(featx,y,'traindata_'+str(iname)+'_'+str(item)+'.scl',zero_based=False)

    featx = []
    allclf = []
    for i in range(1):
        Xs = X
        Xts = Xt

        clf = SVC(random_state=0, probability=True)
        clf.fit(Xs, y)
        allclf.append(clf)
        pr = clf.predict_proba(Xts)
        feat = pr
        if (i == 0):
            featx = feat
        else:
            featx = np.concatenate((featx,feat),axis=1)

        clf = RandomForestClassifier(random_state=0)
        clf.fit(Xs, y)
        allclf.append(clf)
        pr = clf.predict_proba(Xts)
        feat = pr
        featx = np.concatenate((featx,feat),axis=1)

        clf = ExtraTreesClassifier(random_state=0)
        clf.fit(Xs, y)
        allclf.append(clf)
        pr = clf.predict_proba(Xts)
        feat = pr
        featx = np.concatenate((featx,feat),axis=1)


        clf = LGBMClassifier(random_state=0)
        clf.fit(Xs, y)
        allclf.append(clf)
        pr = clf.predict_proba(Xts)
        feat = pr
        featx = np.concatenate((featx,feat),axis=1)

        clf = MLPClassifier(random_state=0)
        clf.fit(Xs, y)
        allclf.append(clf)
        pr = clf.predict_proba(Xts)
        feat = pr
        featx = np.concatenate((featx,feat),axis=1)

        clf = LogisticRegression(random_state=0,  max_iter=10000)
        clf.fit(Xs, y)
        allclf.append(clf)
        pr = clf.predict_proba(Xts)
        feat = pr
        featx = np.concatenate((featx,feat),axis=1)

        clf = OneVsRestClassifier(PLS())
        clf.fit(Xs.toarray(), y)
        allclf.append(clf)
        pr = clf.predict_proba(Xts.toarray())
        feat = pr
        featx = np.concatenate((featx,feat),axis=1)

    from sklearn.datasets import dump_svmlight_file
    dump_svmlight_file(featx,yt,'testdata_'+str(iname)+'_'+str(item)+'.scl',zero_based=False)


