import numpy as np
import pandas as pd

import os
import joblib
model_path = os.path.dirname(os.getcwd()) + '\\model\\'
feat1, yall = joblib.load(model_path+"text_bow1_kt_fs.pkl")
feat2 = joblib.load(model_path+"text_bow2_kt_fs.pkl")[0]
feat3 = joblib.load(model_path+"text_tfidf1_kt_fs.pkl")[0]
feat4 = joblib.load(model_path+"text_tfidf2_kt_fs.pkl")[0]
feat5 = joblib.load(model_path+"text_dict_bow1_kt.pkl")[0]
feat6 = joblib.load(model_path+"text_dict_bow2_kt.pkl")[0]
feat7 = joblib.load(model_path+"text_dict_tfidf1_kt.pkl")[0]
feat8 = joblib.load(model_path+"text_dict_tfidf2_kt.pkl")[0]
feat9 = joblib.load(model_path+"text_w2v_tfidf_kt.pkl")[0]
feat10 = joblib.load(model_path+"text_pos_bow1_kt.pkl")[0]

fname = ["BOW1","BOW2", "TFIDF1", "TFIDF2", "DICTBOW1", "DICTBOW2", "DICTTFIDF1", "DICTTFIDF2", "W2V", "POSTAG"]
stackfname = "KNN,LR,NB,SVM,RF,ET".split()

from sklearn.model_selection import train_test_split
[f1,ft1, f2, ft2, f3, ft3, f4, ft4, f5, ft5, f6, ft6, f7, ft7, f8, ft8, f9, ft9, f10, ft10, y, yt] = train_test_split(feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10, yall, test_size=0.2, random_state=0)
Xall = [eval('f%d' % (i)) for i in range(1,11)]
Xtall = [eval('ft%d' % (i)) for i in range(1,11)]

from sklearn.preprocessing import MaxAbsScaler
allscaler = []
Xalls = []
Xtalls = []
for i in range(0, 9):
    scaler = MaxAbsScaler()
    scaler.fit(Xall[i])
    Xalls.append(scaler.transform(Xall[i]))
    Xtalls.append(scaler.transform(Xtall[i]))
    allscaler.append(scaler)
    
#Save Scaler
from joblib import dump
dump(scaler, "scalews_fs_raw.sav")

# KNN, LR, NB, SVM, RF, ET
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

r = Xalls[0].shape[0]
rt = Xtalls[0].shape[0]
y = y.todense()
y = np.array(y.reshape(r,))[0]
yt = yt.todense()
yt = np.array(yt.reshape(rt,))[0]
ix = []
nr_fold = 10
for i in range(0, r):
    ix.append(i)
ix = np.array(ix)

for i in range(0,len(Xalls)):
    feat = np.zeros((r,),dtype=float)
    feat1 = np.zeros((r,),dtype=float)
    feat2 = np.zeros((r,),dtype=float)
    for j in range(0, nr_fold):
        Xs = Xalls[i]
        train_ix = ((ix % nr_fold) != j)
        test_ix = ((ix % nr_fold) == j)
        train_X, test_X = Xs[train_ix], Xs[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(train_X, train_y)
        pr = clf.predict_proba(test_X)[:,0]
        feat[test_ix] = pr
        pr1 = clf.predict_proba(test_X)[:,1]
        feat1[test_ix] = pr1
        pr2 = clf.predict_proba(test_X)[:,2]
        feat2[test_ix] = pr2
    feat = np.reshape(feat,(r,1))
    feat1 = np.reshape(feat1,(r,1))
    feat2 = np.reshape(feat2,(r,1))
    if (i == 0):
        featx = feat
        featx1 = feat1
        featx2 = feat2
    else:
        featx = np.concatenate((featx,feat),axis=1)
        featx1 = np.concatenate((featx1,feat1),axis=1)
        featx2 = np.concatenate((featx2,feat2),axis=1)

    feat = np.zeros((r,),dtype=float)
    feat1 = np.zeros((r,),dtype=float)
    feat2 = np.zeros((r,),dtype=float)
    for j in range(0, nr_fold):
        Xs = Xalls[i]
        train_ix = ((ix % nr_fold) != j)
        test_ix = ((ix % nr_fold) == j)
        train_X, test_X = Xs[train_ix], Xs[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf = LogisticRegression(random_state=0, max_iter=5000)
        clf.fit(train_X, train_y)
        pr = clf.predict_proba(test_X)[:,0]
        feat[test_ix] = pr
        pr1 = clf.predict_proba(test_X)[:,1]
        feat1[test_ix] = pr1
        pr2 = clf.predict_proba(test_X)[:,2]
        feat2[test_ix] = pr2
    feat = np.reshape(feat,(r,1))
    feat1 = np.reshape(feat1,(r,1))
    feat2 = np.reshape(feat2,(r,1))
    featx = np.concatenate((featx,feat),axis=1)
    featx1 = np.concatenate((featx1,feat1),axis=1)
    featx2 = np.concatenate((featx2,feat2),axis=1)
        
    feat = np.zeros((r,),dtype=float)
    feat1 = np.zeros((r,),dtype=float)
    feat2 = np.zeros((r,),dtype=float)
    for j in range(0, nr_fold):
        Xs = Xalls[i]
        train_ix = ((ix % nr_fold) != j)
        test_ix = ((ix % nr_fold) == j)
        train_X, test_X = Xs[train_ix], Xs[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf = MultinomialNB()
        clf.fit(train_X, train_y)
        pr = clf.predict_proba(test_X)[:,0]
        feat[test_ix] = pr
        pr1 = clf.predict_proba(test_X)[:,1]
        feat1[test_ix] = pr1
        pr2 = clf.predict_proba(test_X)[:,2]
        feat2[test_ix] = pr2
    feat = np.reshape(feat,(r,1))
    feat1 = np.reshape(feat1,(r,1))
    feat2 = np.reshape(feat2,(r,1))
    featx = np.concatenate((featx,feat),axis=1)
    featx1 = np.concatenate((featx1,feat1),axis=1)
    featx2 = np.concatenate((featx2,feat2),axis=1)
    
    feat = np.zeros((r,),dtype=float)
    feat1 = np.zeros((r,),dtype=float)
    feat2 = np.zeros((r,),dtype=float)
    for j in range(0, nr_fold):
        Xs = Xalls[i]
        train_ix = ((ix % nr_fold) != j)
        test_ix = ((ix % nr_fold) == j)
        train_X, test_X = Xs[train_ix], Xs[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf = SVC(probability=True, random_state=0)
        clf.fit(train_X, train_y)
        pr = clf.predict_proba(test_X)[:,0]
        feat[test_ix] = pr
        pr1 = clf.predict_proba(test_X)[:,1]
        feat1[test_ix] = pr1
        pr2 = clf.predict_proba(test_X)[:,2]
        feat2[test_ix] = pr2
    feat = np.reshape(feat,(r,1))
    feat1 = np.reshape(feat1,(r,1))
    feat2 = np.reshape(feat2,(r,1))
    featx = np.concatenate((featx,feat),axis=1)
    featx1 = np.concatenate((featx1,feat1),axis=1)
    featx2 = np.concatenate((featx2,feat2),axis=1)
    
    feat = np.zeros((r,),dtype=float)
    feat1 = np.zeros((r,),dtype=float)
    feat2 = np.zeros((r,),dtype=float)
    for j in range(0, nr_fold):
        Xs = Xalls[i]
        train_ix = ((ix % nr_fold) != j)
        test_ix = ((ix % nr_fold) == j)
        train_X, test_X = Xs[train_ix], Xs[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf = RandomForestClassifier(random_state=0)
        clf.fit(train_X, train_y)
        pr = clf.predict_proba(test_X)[:,0]
        feat[test_ix] = pr
        pr1 = clf.predict_proba(test_X)[:,1]
        feat1[test_ix] = pr1
        pr2 = clf.predict_proba(test_X)[:,2]
        feat2[test_ix] = pr2
    feat = np.reshape(feat,(r,1))
    feat1 = np.reshape(feat1,(r,1))
    feat2 = np.reshape(feat2,(r,1))
    featx = np.concatenate((featx,feat),axis=1)
    featx1 = np.concatenate((featx1,feat1),axis=1)
    featx2 = np.concatenate((featx2,feat2),axis=1)
    
    feat = np.zeros((r,),dtype=float)
    feat1 = np.zeros((r,),dtype=float)
    feat2 = np.zeros((r,),dtype=float)
    for j in range(0, nr_fold):
        Xs = Xalls[i]
        train_ix = ((ix % nr_fold) != j)
        test_ix = ((ix % nr_fold) == j)
        train_X, test_X = Xs[train_ix], Xs[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf = ExtraTreesClassifier(random_state=0)
        clf.fit(train_X, train_y)
        pr = clf.predict_proba(test_X)[:,0]
        feat[test_ix] = pr
        pr1 = clf.predict_proba(test_X)[:,1]
        feat1[test_ix] = pr1
        pr2 = clf.predict_proba(test_X)[:,2]
        feat2[test_ix] = pr2
    feat = np.reshape(feat,(r,1))
    feat1 = np.reshape(feat1,(r,1))
    feat2 = np.reshape(feat2,(r,1))
    featx = np.concatenate((featx,feat),axis=1)
    featx1 = np.concatenate((featx1,feat1),axis=1)
    featx2 = np.concatenate((featx2,feat2),axis=1)
    
from sklearn.preprocessing import MinMaxScaler
X_final = np.hstack((featx,featx1,featx2))
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_final)
X_train_norm =  scaler.transform(X_final)

#Save Scaler
from joblib import dump
dump(scaler, "scalews_fs_stack.sav")

from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
dump_svmlight_file(X_train_norm,y,'traindata_ws_fs.scl',zero_based=False)

allclf = []  
for i in range(0,len(Xalls)):
    Xs = Xalls[i]
    Xts = Xtalls[i]
    
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(Xs, y)
    allclf.append(clf)
    pr = clf.predict_proba(Xts)[:,0]
    pr1 = clf.predict_proba(Xts)[:,1]
    pr2 = clf.predict_proba(Xts)[:,2]
    feat = pr
    feat = np.reshape(feat,(rt,1))
    feat1 = pr1
    feat1 = np.reshape(feat1,(rt,1))
    feat2 = pr2
    feat2 = np.reshape(feat2,(rt,1))
    if (i == 0):
        featx = feat
        featx1 = feat1
        featx2 = feat2
    else:
        featx = np.concatenate((featx,feat),axis=1)
        featx1 = np.concatenate((featx1,feat1),axis=1)
        featx2 = np.concatenate((featx2,feat2),axis=1)
    
    clf = LogisticRegression(random_state=0, max_iter=5000)
    clf.fit(Xs, y)
    allclf.append(clf)
    pr = clf.predict_proba(Xts)[:,0]
    pr1 = clf.predict_proba(Xts)[:,1]
    pr2 = clf.predict_proba(Xts)[:,2]
    
    feat = pr
    feat = np.reshape(feat,(rt,1))
    feat1 = pr1
    feat1 = np.reshape(feat1,(rt,1))
    feat2 = pr2
    feat2 = np.reshape(feat2,(rt,1))
    
    featx = np.concatenate((featx,feat),axis=1)
    featx1 = np.concatenate((featx1,feat1),axis=1)
    featx2 = np.concatenate((featx2,feat2),axis=1)
    
    clf = MultinomialNB()
    clf.fit(Xs, y)
    allclf.append(clf)
    pr = clf.predict_proba(Xts)[:,0]
    pr1 = clf.predict_proba(Xts)[:,1]
    pr2 = clf.predict_proba(Xts)[:,2]
    
    feat = pr
    feat = np.reshape(feat,(rt,1))
    feat1 = pr1
    feat1 = np.reshape(feat1,(rt,1))
    feat2 = pr2
    feat2 = np.reshape(feat2,(rt,1))
    
    featx = np.concatenate((featx,feat),axis=1)
    featx1 = np.concatenate((featx1,feat1),axis=1)
    featx2 = np.concatenate((featx2,feat2),axis=1)
    
    clf = SVC(probability=True, random_state=0)
    clf.fit(Xs, y)
    allclf.append(clf)
    pr = clf.predict_proba(Xts)[:,0]
    pr1 = clf.predict_proba(Xts)[:,1]
    pr2 = clf.predict_proba(Xts)[:,2]
    
    feat = pr
    feat = np.reshape(feat,(rt,1))
    feat1 = pr1
    feat1 = np.reshape(feat1,(rt,1))
    feat2 = pr2
    feat2 = np.reshape(feat2,(rt,1))
    
    featx = np.concatenate((featx,feat),axis=1)
    featx1 = np.concatenate((featx1,feat1),axis=1)
    featx2 = np.concatenate((featx2,feat2),axis=1)
    
    clf = RandomForestClassifier(random_state=0)
    clf.fit(Xs, y)
    allclf.append(clf)
    pr = clf.predict_proba(Xts)[:,0]
    pr1 = clf.predict_proba(Xts)[:,1]
    pr2 = clf.predict_proba(Xts)[:,2]
    
    feat = pr
    feat = np.reshape(feat,(rt,1))
    feat1 = pr1
    feat1 = np.reshape(feat1,(rt,1))
    feat2 = pr2
    feat2 = np.reshape(feat2,(rt,1))
    
    featx = np.concatenate((featx,feat),axis=1)
    featx1 = np.concatenate((featx1,feat1),axis=1)
    featx2 = np.concatenate((featx2,feat2),axis=1)

    clf = ExtraTreesClassifier(random_state=0)
    clf.fit(Xs, y)
    allclf.append(clf)
    pr = clf.predict_proba(Xts)[:,0]
    pr1 = clf.predict_proba(Xts)[:,1]
    pr2 = clf.predict_proba(Xts)[:,2]
    
    feat = pr
    feat = np.reshape(feat,(rt,1))
    feat1 = pr1
    feat1 = np.reshape(feat1,(rt,1))
    feat2 = pr2
    feat2 = np.reshape(feat2,(rt,1))
    
    featx = np.concatenate((featx,feat),axis=1)
    featx1 = np.concatenate((featx1,feat1),axis=1)
    featx2 = np.concatenate((featx2,feat2),axis=1)

Xt_final = np.hstack((featx,featx1,featx2))
X_test_norm =  scaler.transform(Xt_final)
dump_svmlight_file(X_test_norm,yt,'testdata_ws_fs.scl',zero_based=False)

# Save Prob Model
from joblib import dump
dump(allclf, "allmodelws_fs.sav")