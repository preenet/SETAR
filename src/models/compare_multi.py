import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef # average == 'macro'.
from sklearn.metrics import roc_auc_score # multiclas 'ovo' average == 'macro'.


def cv(clf, X, y, nr_fold):
    ix = []
    for i in range(0, len(y)):
        ix.append(i)
    ix = np.array(ix)
    
    allACC = []
    allSENS = []
    allSPEC = []
    allMCC = []
    allAUC = []
    for j in range(0, nr_fold):
        train_ix = ((ix % nr_fold) != j)
        test_ix = ((ix % nr_fold) == j)
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf.fit(train_X, train_y)        
        p = clf.predict(test_X)
        pr = clf.predict_proba(test_X)  
        
        ACC = accuracy_score(test_y,p)
        SENS = precision_score(test_y,p, average='macro')
        SPEC = recall_score(test_y,p, average='macro')
        MCC = matthews_corrcoef(test_y,p)
        AUC = roc_auc_score(test_y,pr,multi_class='ovo',average='macro')
        allACC.append(ACC)
        allSENS.append(SENS)
        allSPEC.append(SPEC)
        allMCC.append(MCC)
        allAUC.append(AUC)
    return np.mean(allACC),np.mean(allSENS),np.mean(allSPEC),np.mean(allMCC),np.mean(allAUC)

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
    
    def fcount(string, substr):
       count = 0
       pos = 0
       while(True):
           pos = string.find(substr , pos)
           if pos > -1:
               count = count + 1
               pos += 1
           else:
               break
       return count
    return ACC, SENS, SPEC, MCC, AUC

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from sklearn.datasets import load_svmlight_file
import joblib
def get_data():
    data = load_svmlight_file("./traindata.scl", zero_based=False)
    data1 = load_svmlight_file("./testdata.scl", zero_based=False)
    return data[0].toarray(), data[1], data1[0].toarray(), data1[1]
X, y, Xt, yt = get_data()
    
    
allclf = []
file = open("11classifier_cv.csv", "w")

#SVM
param = [1,2,4,8,16,32]
acc = np.zeros(len(param)) 
sens = np.zeros(len(param)) 
spec = np.zeros(len(param)) 
mcc = np.zeros(len(param)) 
roc = np.zeros(len(param)) 
for i in range(0,len(param)):
    clf = SVC(C=param[i], random_state=0, probability=True)
    acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
choose = np.argmax(acc)
allclf.append(SVC(C=param[choose], random_state=0, probability=True).fit(X,y))
file.write("SVM,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

#LinearSVC
param = [1,2,4,8,16,32]
acc = np.zeros(len(param)) 
sens = np.zeros(len(param)) 
spec = np.zeros(len(param)) 
mcc = np.zeros(len(param)) 
roc = np.zeros(len(param)) 
for i in range(0,len(param)):
    clf =  SVC(C=param[i], kernel='linear',random_state=0, probability=True)
    acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
choose = np.argmax(acc)
allclf.append(SVC(C=param[i], kernel='linear',random_state=0, probability=True).fit(X,y))
file.write("LN,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

#RF
param = [20, 50, 100, 200]
acc = np.zeros(len(param)) 
sens = np.zeros(len(param)) 
spec = np.zeros(len(param)) 
mcc = np.zeros(len(param)) 
roc = np.zeros(len(param)) 
for i in range(0,len(param)):
    clf = RandomForestClassifier(n_estimators=param[i], random_state=0)
    acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
choose = np.argmax(acc)
allclf.append(RandomForestClassifier(n_estimators=param[choose], random_state=0).fit(X,y))
file.write("RF,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

#E-Tree
param = [20, 50, 100, 200]
acc = np.zeros(len(param)) 
sens = np.zeros(len(param)) 
spec = np.zeros(len(param)) 
mcc = np.zeros(len(param)) 
roc = np.zeros(len(param)) 
for i in range(0,len(param)):
    clf = ExtraTreesClassifier(n_estimators=param[i], random_state=0)
    acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
choose = np.argmax(acc)
allclf.append(ExtraTreesClassifier(n_estimators=param[choose], random_state=0).fit(X,y))
file.write("ET,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

#XGBoost
param = [20, 50, 100, 200]
acc = np.zeros(len(param)) 
sens = np.zeros(len(param)) 
spec = np.zeros(len(param)) 
mcc = np.zeros(len(param)) 
roc = np.zeros(len(param)) 
for i in range(0,len(param)):
    clf = XGBClassifier(n_estimators=param[i],learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=0)
    acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
choose = np.argmax(acc)  
allclf.append(XGBClassifier(n_estimators=param[i],learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=0).fit(X,y))
file.write("XGB,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

#LightGBM
param = [20, 50, 100, 200]
acc = np.zeros(len(param)) 
sens = np.zeros(len(param)) 
spec = np.zeros(len(param)) 
mcc = np.zeros(len(param)) 
roc = np.zeros(len(param)) 
for i in range(0,len(param)):
    clf = LGBMClassifier(n_estimators=param[i],learning_rate=0.1, random_state=0)
    acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
choose = np.argmax(acc)  
allclf.append(LGBMClassifier(n_estimators=param[i],learning_rate=0.1, random_state=0).fit(X,y))
file.write("LGBM,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

#MLP
param = [20, 50, 100, 200]
acc = np.zeros(len(param)) 
sens = np.zeros(len(param)) 
spec = np.zeros(len(param)) 
mcc = np.zeros(len(param)) 
roc = np.zeros(len(param)) 
for i in range(0,len(param)):  
    clf = MLPClassifier(hidden_layer_sizes=(param[i],),random_state=0, max_iter = 10000)
    acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
choose = np.argmax(acc)
allclf.append(MLPClassifier(hidden_layer_sizes=(param[choose],),random_state=0, max_iter=10000).fit(X,y))
file.write("MLP,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n") 

#NB
clf = GaussianNB()
acc, sens, spec, mcc, roc = cv(clf, X,y,10) 
allclf.append(clf)
file.write("NB,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n") 

#1NN
clf = KNeighborsClassifier(n_neighbors=1)
acc, sens, spec, mcc, roc = cv(clf, X,y,10) 
allclf.append(clf)
file.write("1NN,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n")

#DT
clf = DecisionTreeClassifier(random_state=0)
acc, sens, spec, mcc, roc = cv(clf, X,y,10) 
allclf.append(clf)
file.write("DT,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n") 

#Logistic
param = [0.001,0.01,0.1,1,10,100]
acc = np.zeros(len(param)) 
sens = np.zeros(len(param)) 
spec = np.zeros(len(param)) 
mcc = np.zeros(len(param)) 
roc = np.zeros(len(param)) 
for i in range(0,len(param)):
    clf = LogisticRegression(C=param[i], random_state=0, max_iter=10000)
    acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
choose = np.argmax(acc)
allclf.append(LogisticRegression(C=param[choose], random_state=0, max_iter=10000).fit(X,y))
file.write("LR,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")   

file.close()

########## Test ############################
file = open("11classifier_test.csv", "w")
for i in range(0,len(allclf)):
    acc, sens, spec, mcc, roc = test(allclf[i], X, y, Xt, yt) 
    file.write(str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+"\n") 
file.close()

