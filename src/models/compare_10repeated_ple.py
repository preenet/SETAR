#df_ds = pd.read_csv('/kaggle/input/processed-wisesight/wisesight.txt')
#y_ds = df_ds['target'].astype('category').cat.codes
#Xo = [' '.join(process_text(item))  for item in df_ds['text']]
#yo = y_ds.to_numpy()
#import joblib
#joblib.dump((Xo, yo), "ws.sav")
import pandas as pd
import src.utilities as utils
from sklearnex import patch_sklearn

config = utils.read_config()
patch_sklearn()
df_ds = pd.read_csv(config['data']['processed_kt'])
y_ds = df_ds['target'].astype('category').cat.codes
Xo = df_ds['processed']
yo = y_ds.to_numpy()
import joblib

joblib.dump((Xo, yo), "ws.sav")
    
import sys

import joblib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import sparse

PATH = "."
Xo, yo = joblib.load(PATH+"/ws.sav")
# Testing Only 
#idx1 = list(np.random.choice(np.where(yo==0)[0], 10, replace=False))
#idx2 = list(np.random.choice(np.where(yo==1)[0], 10, replace=False))
#idx3 = list(np.random.choice(np.where(yo==2)[0], 10, replace=False))
#idx = np.array(idx1+idx2+idx3)
#Xo = np.array(Xo)
#Xo = Xo[idx]
#yo = yo[idx]
SEED = [i for i in range(0,10)]
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
file.write('        pr = [np.round(np.abs(item[0])) for item in self.clf.predict(X)]'+"\n")
file.write('        return np.array(pr)'+"\n")
file.write('    def predict_proba(self, X):'+"\n")
file.write('        p_all = []'+"\n")
file.write('        p_all.append([1-np.abs(item[0]) for item in self.clf.predict(X)])'+"\n")
file.write('        p_all.append([np.abs(item[0]) for item in self.clf.predict(X)])'+"\n")
file.write('        return np.transpose(np.array(p_all))'+"\n")
file.close()

import gensim
from lightgbm import LGBMClassifier
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
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from PLS import PLS


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

# posdict = pd.read_csv('./pos.txt', header=None)[0].tolist()
# negdict = pd.read_csv('./neg.txt', header=None)[0].tolist()
# dict = np.unique(posdict + negdict)

from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingVectorizer(object):
    def __init__(self, type="mean", size=300, window=5, min_count=1, seed=0):
        self.type = type
        self.dim = size
        self.window = window
        self.min_count = min_count
        self.seed = seed
        self.word2vec = None
        self.word2weight = None

    def fit(self, X):
        tok_train = [text.split() for text in X]
        model = gensim.models.Word2Vec(tok_train, size=self.dim, window=self.window, min_count=self.min_count,seed=self.seed, iter=100)
        self.word2vec = {w: vec for w, vec in zip(model.wv.index2word, model.wv.vectors)}
        self.word2weight = None
        self.dim = model.vector_size

        if self.type == "tfidf":
            tfidf = TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=(1, 1), min_df=self.min_count, sublinear_tf=True)
            tfidf.fit(X)
            max_idf = max(tfidf.idf_)
            self.word2weight = defaultdict(lambda: max_idf,[(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        if self.type == "tfidf":
            return sparse.csr_matrix([np.mean([self.word2vec[w] * self.word2weight[w] for w in words.split() if w in self.word2vec] or [np.zeros(self.dim)], axis=0) for words in X], dtype="float32")
        else:
            return sparse.csr_matrix([np.mean([self.word2vec[w] for w in words.split() if w in self.word2vec] or [np.zeros(self.dim)], axis=0) for words in X], dtype="float32")

#SEED = int(sys.argv[-2])
for item in SEED:
    X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=item, stratify=yo)
    X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)
    iname = sys.argv[-1]
    fe = {'BOW1': CountVectorizer(tokenizer=lambda x:x.split(), ngram_range=(1, 1), min_df=20),
          'BOW12': CountVectorizer(tokenizer=lambda x:x.split(), ngram_range=(1, 2), min_df=20),
          'BOW2': CountVectorizer(tokenizer=lambda x:x.split(), ngram_range=(2, 2)),
          'TF1': TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=(1, 1), min_df=20, sublinear_tf=True),
          'TF12': TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=(1, 2), min_df=20, sublinear_tf=True),
          'TF2': TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=(2, 2), min_df=20, sublinear_tf=True),
          'DICTBOW': CountVectorizer(vocabulary=dict, tokenizer=lambda x:x.split(), ngram_range=(1, 1)),
          'DICTTF': TfidfVectorizer(vocabulary=dict, tokenizer=lambda x:x.split(), ngram_range=(1, 1), sublinear_tf=True),
          'W2V': EmbeddingVectorizer(),
          'W2VTF': EmbeddingVectorizer('tfidf')}
    fe[iname].fit(X_train)
    X_train_val = fe[iname].transform(X_train)
    X_val_val = fe[iname].transform(X_val)
    X_test_val = fe[iname].transform(X_test)
    scaler = MaxAbsScaler()
    scaler.fit(X_train_val)
    X =  scaler.transform(X_train_val)
    Xv = scaler.transform(X_val_val)
    Xt = scaler.transform(X_test_val)
    allclf = []
    file = open("12classifier_"+iname+str(SEED)+"_res.csv", "a")

    #SVM
    param = [1,2,4,8,16,32]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    f1 = np.zeros(len(param))
    for i in range(0,len(param)):
        clf = SVC(C=param[i], random_state=0, probability=True)
        acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)
    allclf.append(SVC(C=param[choose], random_state=0, probability=True).fit(X,y))
    file.write(str(item)+"SVMRBF,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

    #LinearSVC
    param = [1,2,4,8,16,32]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param))
    f1 = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf =  SVC(C=param[i], kernel='linear',random_state=0, probability=True)
        acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)
    allclf.append(SVC(C=param[i], kernel='linear',random_state=0, probability=True).fit(X,y))
    file.write(str(item)+"SVMLN,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

    #RF
    param = [20, 50, 100, 200]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    f1 = np.zeros(len(param))
    for i in range(0,len(param)):
        clf = RandomForestClassifier(n_estimators=param[i], random_state=0)
        acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)
    allclf.append(RandomForestClassifier(n_estimators=param[choose], random_state=0).fit(X,y))
    file.write(str(item)+"RF,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

    #E-Tree
    param = [20, 50, 100, 200]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    f1 = np.zeros(len(param))
    for i in range(0,len(param)):
        clf = ExtraTreesClassifier(n_estimators=param[i], random_state=0)
        acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)
    allclf.append(ExtraTreesClassifier(n_estimators=param[choose], random_state=0).fit(X,y))
    file.write(str(item)+"ET,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

    #XGBoost
    param = [20, 50, 100, 200]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    f1 = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf = XGBClassifier(n_estimators=param[i],learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=0)
        acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)  
    allclf.append(XGBClassifier(n_estimators=param[i],learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=0).fit(X,y))
    file.write(str(item)+"XGB,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

    #LightGBM
    param = [20, 50, 100, 200]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    f1 = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf = LGBMClassifier(n_estimators=param[i],learning_rate=0.1, random_state=0)
        acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)  
    allclf.append(LGBMClassifier(n_estimators=param[i],learning_rate=0.1, random_state=0).fit(X,y))
    file.write(str(item)+"LGBM,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

    #MLP
    param = [20, 50, 100, 200]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    f1 = np.zeros(len(param))
    for i in range(0,len(param)):  
        clf = MLPClassifier(hidden_layer_sizes=(param[i],),random_state=0, max_iter = 10000)
        acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)
    allclf.append(MLPClassifier(hidden_layer_sizes=(param[choose],),random_state=0, max_iter=10000).fit(X,y))
    file.write(str(item)+"MLP,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose])) 
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

    #NB
    clf = GaussianNB()
    acc, sens, spec, mcc, roc, f1 = test(clf,X.toarray(),y,Xv.toarray(),yv)
    allclf.append(clf)
    file.write(str(item)+"NB,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA"))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

    #1NN
    clf = KNeighborsClassifier(n_neighbors=1)
    acc, sens, spec, mcc, roc, f1 = test(clf,X,y,Xv,yv)
    allclf.append(clf)
    file.write(str(item)+"1NN,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA"))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

    #DT
    clf = DecisionTreeClassifier(random_state=0)
    acc, sens, spec, mcc, roc, f1 = test(clf,X,y,Xv,yv)
    allclf.append(clf)
    file.write(str(item)+"DT,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA")) 
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

    #Logistic
    param = [0.001,0.01,0.1,1,10,100]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    f1 = np.zeros(len(param))
    for i in range(0,len(param)):
        clf = LogisticRegression(C=param[i], random_state=0, max_iter=10000)
        acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)
    allclf.append(LogisticRegression(C=param[choose], random_state=0, max_iter=10000).fit(X,y))
    file.write(str(item)+"LR,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))   
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

    #PLS
    clf = OneVsRestClassifier(PLS())
    acc, sens, spec, mcc, roc, f1 = test(clf,X.toarray(),y,Xv.toarray(),yv)
    allclf.append(clf)
    file.write(str(item)+"PLS,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA"))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

    file.close()
