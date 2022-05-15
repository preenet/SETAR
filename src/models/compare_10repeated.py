import src.utilities as utils
from src.feature.process_thai_text import process_text
import pandas as pd
config = utils.read_config()

df_ds = pd.read_csv(config['data']['processed_ws'])
y_ds = df_ds['target'].astype('category').cat.codes

Xo = [' '.join(process_text(item))  for item in df_ds['text']]
yo = y_ds.to_numpy()

import joblib
joblib.dump((Xo, yo), "ws.sav")
from scipy import sparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib
PATH = "."
Xo, yo = joblib.load(PATH+"/ws.sav")
SEED = [i for i in range(0,10)]


from PLS import PLS
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef # average == 'macro'.
from sklearn.metrics import roc_auc_score # multiclas 'ovo' average == 'macro'.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
    return ACC, SENS, SPEC, MCC, AUC
    
def testPLS(clf, X, y, Xt, yt):
    train_X, test_X = X.toarray(), Xt.toarray()
    train_y, test_y = y, yt
    clf.fit(train_X, train_y)        
    p = clf.predict(test_X)
    pr = clf.predict_proba(test_X)
    ACC = accuracy_score(test_y,p)
    SENS = precision_score(test_y,p, average='macro')
    SPEC = recall_score(test_y,p, average='macro')
    MCC = matthews_corrcoef(test_y,p)
    AUC = roc_auc_score(test_y,pr,multi_class='ovo',average='macro')
    return ACC, SENS, SPEC, MCC, AUC

for item in SEED:
    X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=item, stratify=yo)
    X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)
    iname = "TFIDF1"
    #fe = CountVectorizer(tokenizer=lambda x:x.split(), ngram_range=(1, 1), min_df=5)
    #fe = CountVectorizer(tokenizer=lambda x:x.split(), ngram_range=(2, 2), min_df=5)
    fe = TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=(1, 1), min_df=20)
    #fe = TfidfVectorizer(tokenizer=lambda x:x.split(), ngram_range=(2, 2), min_df=5)

    fe.fit(X_train)
    X_train_val = sparse.csr_matrix(fe.transform(X_train).toarray())
    X_val_val = sparse.csr_matrix(fe.transform(X_val).toarray())
    X_test_val = sparse.csr_matrix(fe.transform(X_test).toarray())
    
    scaler = MaxAbsScaler()
    scaler.fit(X_train_val)
    X = scaler.transform(X_train_val)
    Xv = scaler.transform(X_val_val)
    Xt = scaler.transform(X_test_val)   
    
    allclf = []
    file = open("12classifier_"+iname+"_val.csv", "a")

    #SVM
    param = [1,2,4,8,16,32]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf = SVC(C=param[i], random_state=0, probability=True)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)
    allclf.append(SVC(C=param[choose], random_state=0, probability=True).fit(X,y))
    file.write(str(item)+"SVM,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

    #LinearSVC
    param = [1,2,4,8,16,32]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf =  SVC(C=param[i], kernel='linear',random_state=0, probability=True)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)
    allclf.append(SVC(C=param[i], kernel='linear',random_state=0, probability=True).fit(X,y))
    file.write(str(item)+"LN,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

    #RF
    param = [20, 50, 100, 200]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf = RandomForestClassifier(n_estimators=param[i], random_state=0)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)
    allclf.append(RandomForestClassifier(n_estimators=param[choose], random_state=0).fit(X,y))
    file.write(str(item)+"RF,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

    #E-Tree
    param = [20, 50, 100, 200]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf = ExtraTreesClassifier(n_estimators=param[i], random_state=0)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)
    allclf.append(ExtraTreesClassifier(n_estimators=param[choose], random_state=0).fit(X,y))
    file.write(str(item)+"ET,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

    #XGBoost
    param = [20, 50, 100, 200]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf = XGBClassifier(n_estimators=param[i],learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=0)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)  
    allclf.append(XGBClassifier(n_estimators=param[i],learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=0).fit(X,y))
    file.write(str(item)+"XGB,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

    #LightGBM
    param = [20, 50, 100, 200]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf = LGBMClassifier(n_estimators=param[i],learning_rate=0.1, random_state=0)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)  
    allclf.append(LGBMClassifier(n_estimators=param[i],learning_rate=0.1, random_state=0).fit(X,y))
    file.write(str(item)+"LGBM,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

    #MLP
    param = [20, 50, 100, 200]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):  
        clf = MLPClassifier(hidden_layer_sizes=(param[i],),random_state=0, max_iter = 10000)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)
    allclf.append(MLPClassifier(hidden_layer_sizes=(param[choose],),random_state=0, max_iter=10000).fit(X,y))
    file.write(str(item)+"MLP,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n") 

    #NB
    clf = MultinomialNB()
    acc, sens, spec, mcc, roc = test(clf,X,y,Xv,yv)
    allclf.append(clf)
    file.write(str(item)+"NB,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n") 

    #1NN
    clf = KNeighborsClassifier(n_neighbors=1)
    acc, sens, spec, mcc, roc = test(clf,X,y,Xv,yv)
    allclf.append(clf)
    file.write(str(item)+"1NN,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n")

    #DT
    clf = DecisionTreeClassifier(random_state=0)
    acc, sens, spec, mcc, roc = test(clf,X,y,Xv,yv)
    allclf.append(clf)
    file.write(str(item)+"DT,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n") 

    #Logistic
    param = [0.001,0.01,0.1,1,10,100]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf = LogisticRegression(C=param[i], random_state=0, max_iter=10000)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = test(clf,X,y,Xv,yv)
    choose = np.argmax(acc)
    allclf.append(LogisticRegression(C=param[choose], random_state=0, max_iter=10000).fit(X,y))
    file.write(str(item)+"LR,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")   

    #PLS
    clf = OneVsRestClassifier(PLS())
    acc, sens, spec, mcc, roc = testPLS(clf,X,y,Xv,yv)
    allclf.append(clf)
    file.write(str(item)+"PLS,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n") 

    file.close()

    ########## Test ############################
    file = open("12classifier_"+iname+"_test.csv", "a")
    for i in range(0,len(allclf)-1):
        acc, sens, spec, mcc, roc = test(allclf[i], X, y, Xt, yt) 
        file.write(str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+"\n") 
    acc, sens, spec, mcc, roc = testPLS(allclf[-1], X, y, Xt, yt) 
    file.write(str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+"\n") 
    file.close()
