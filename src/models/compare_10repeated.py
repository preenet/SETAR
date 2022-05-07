import numpy as np # linear algebra
import sys
import datetime
import src.utilities as utils
import joblib

config = utils.read_config()
PATH = config['data']['final_original']

feat1, yall = joblib.load(PATH+"/text_bow1_kt.pkl")
feat2 = joblib.load(PATH+"/text_bow2_kt.pkl")[0]
feat3 = joblib.load(PATH+"/text_tfidf1_kt.pkl")[0]
feat4 = joblib.load(PATH+"/text_tfidf2_kt.pkl")[0]
feat5 = joblib.load(PATH+"/text_dict_bow1_kt.pkl")[0]
feat6 = joblib.load(PATH+"/text_dict_bow2_kt.pkl")[0]
feat7 = joblib.load(PATH+"/text_dict_tfidf1_kt.pkl")[0]
feat8 = joblib.load(PATH+"/text_dict_tfidf2_kt.pkl")[0]
feat9 = joblib.load(PATH+"/text_w2v_tfidf_kt.pkl")[0]
feat10 = joblib.load(PATH+"/text_pos_bow1_kt.pkl")[0]

fname = ["BOW1","BOW2", "TFIDF1", "TFIDF2", "DICTBOW1", "DICTBOW2", "DICTTFIDF1", "DICTTFIDF2", "W2V", "POSTAG"]
iname = sys.argv[-1]
SEED = [i for i in range(0,10)]
fi = fname.index(iname) + 1

file = open(config['models']+'PLS.py','w')
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
file.write('        return np.transpose(np.nan_to_num(np.array(p_all)))'+"\n")
file.close()

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

yo = yall.toarray().reshape(1,-1)[0]
begin_script_time = datetime.datetime.now()
print("start at: ", begin_script_time)

for item in SEED:
    print("SEED:", item)
    X_train, X_tmp, y, y_tmp = train_test_split(eval('feat%d' % (fi)), yo, test_size=0.4, random_state=item, stratify=yo)
    X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)

    scaler = MaxAbsScaler()
    scaler.fit(X_train)
    X = scaler.transform(X_train)
    Xv = scaler.transform(X_val)
    Xt = scaler.transform(X_test)   
        
    allclf = []
    file = open(config['output']+"12classifier_"+iname+"_val.csv", "a")

    #SVM
    print("Running SVM...")
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
    print("Running LinearSVC...")
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
    print("Running RF...")
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
    print("Running Etree...")
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
    print("Running XGBoost...")
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
    print("Running LightGBM...")
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
    print("Running MLP...")
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
    print("Running NB...")
    clf = MultinomialNB()
    acc, sens, spec, mcc, roc = test(clf,X,y,Xv,yv)
    allclf.append(clf)
    file.write(str(item)+"NB,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n") 

    #1NN
    print("Running 1NN...")
    clf = KNeighborsClassifier(n_neighbors=1)
    acc, sens, spec, mcc, roc = test(clf,X,y,Xv,yv)
    allclf.append(clf)
    file.write(str(item)+"1NN,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n")

    #DT
    print("Running DT...")
    clf = DecisionTreeClassifier(random_state=0)
    acc, sens, spec, mcc, roc = test(clf,X,y,Xv,yv)
    allclf.append(clf)
    file.write(str(item)+"DT,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n") 

    #Logistic
    print("Running Logistic...")
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
    print("Running PLS...")
    clf = OneVsRestClassifier(PLS())
    acc, sens, spec, mcc, roc = testPLS(clf,X,y,Xv,yv)
    allclf.append(clf)
    file.write(str(item)+"PLS,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n") 

    file.close()

    ########## Test ############################
    file = open(config['output']+"12classifier_"+iname+"_test.csv", "a")
    for i in range(0,len(allclf)-1):
        acc, sens, spec, mcc, roc = test(allclf[i], X, y, Xt, yt) 
        file.write(str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+"\n") 
    acc, sens, spec, mcc, roc = testPLS(allclf[-1], X, y, Xt, yt) 
    file.write(str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+"\n") 
    file.close()
    print(print("Seed time: ", datetime.datetime.now() - begin_script_time))
print(print("Total time: ", datetime.datetime.now() - begin_script_time))