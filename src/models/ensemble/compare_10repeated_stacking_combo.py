import sys
from pathlib import Path

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import src.utilities as utils

SEED = [i for i in range(8, 10)]


from daal4py.sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              StackingClassifier)
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
from sklearnex import patch_sklearn
from src.models.PLS import PLS
from xgboost import XGBClassifier

configs = utils.read_config()
root = utils.get_project_root()

##################################################################
data_path = str(Path.joinpath(root, configs['data']['wangcha_ws']))

out_file_name = 'ws_combo.csv' 
num_class = 4
##################################################################

patch_sklearn()

def feature_selection(X, y):
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    selector = SelectKBest(mutual_info_classif, k=45)
    X_new = selector.fit_transform(X, y)
    return X_new

def get_stacking():
    ''' 
    return: list of models for level 0
    '''
    level0 = list()
    level0.append(('svm', SVC(random_state=0, probability=True)))
    level0.append(('pls', OneVsRestClassifier(PLS())))
    level0.append(('mlp', MLPClassifier(random_state=0, max_iter=10000)))
    level0.append(('xgb', XGBClassifier(learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=0)))
    level0.append(('rf', RandomForestClassifier(random_state=0)))
    level0.append(('lgbm', LGBMClassifier()))
    level0.append(('et', ExtraTreesClassifier(random_state=0)))
    level0.append(('nb' , GaussianNB()))
    
    # level0.append(('lr' , LogisticRegression(random_state=0, max_iter=10000)))
    # level0.append(('1nn' , KNeighborsClassifier(n_neighbors=1)))
    
    # level0.append(('dt' , DecisionTreeClassifier()))

    return level0

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
    if num_class > 2:
        AUC = roc_auc_score(test_y,pr,multi_class='ovo',average='macro')
    else:
        AUC = roc_auc_score(test_y,pr[:,1]) # for binary classification problem
    F1 = 2*SENS*SPEC/(SENS+SPEC)
    return ACC, SENS, SPEC, MCC, AUC, F1


from sklearn.datasets import load_svmlight_file


def get_data(idx):
    data = load_svmlight_file(data_path + "\\" + "traindata_"+str(idx)+".scl", zero_based=False)
    data1 = load_svmlight_file(data_path + "\\" + "testdata_"+str(idx)+".scl", zero_based=False)
    return data[0].toarray(), data[1], data1[0].toarray(), data1[1]

iname = "WC-STACKING-CB-Ranking"

for item in SEED:
    file = open(configs['output_scratch_propose']+"12classifier_"+iname+"_res_" + out_file_name, "a")
    print("SEED:", item)
    file.write("SEED"+ str(item) +"\n")
    for j in range(1, len(get_stacking())):
       
        print("using:" , j+1 , "combo(s).")
        file.write("using"+ str(j+1) +"combo(s)\n")

        Xa, ya, Xt, yt = get_data(item)
        # Xa = feature_selection(Xa, ya)
        # Xt = feature_selection(Xt, yt)
        
        idx = int(round(len(ya)*0.75))
        X = Xa[0:idx, :]
        y = ya[0:idx]
        Xv = Xa[idx:-1, :]
        yv = ya[idx:-1]

        allclf = []
        
        print("Stacking-SVM...")
        #SVM
        param = [1,2,4,8,16,32]
        acc = np.zeros(len(param)) 
        sens = np.zeros(len(param)) 
        spec = np.zeros(len(param)) 
        mcc = np.zeros(len(param)) 
        roc = np.zeros(len(param)) 
        f1 = np.zeros(len(param))
        for i in range(0,len(param)):
            level0 = get_stacking()[:j+1]
            level1 = SVC(C=param[i], random_state=0, probability=True)
            clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
        choose = np.argmax(acc)
        allclf.append(SVC(C=param[choose], random_state=0, probability=True).fit(X,y))
        file.write(str(item)+"SVMRBF,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
        print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
        print("test_acc:", str(acc), ", test_f1:", str(f1))
        
        #LinearSVC
        print("Stacking-LinearSVC...")
        param = [1,2,4,8,16,32]
        acc = np.zeros(len(param)) 
        sens = np.zeros(len(param)) 
        spec = np.zeros(len(param)) 
        mcc = np.zeros(len(param)) 
        roc = np.zeros(len(param))
        f1 = np.zeros(len(param)) 
        for i in range(0,len(param)):
            level0 =  get_stacking()[:j+1]
            level1 =  SVC(C=param[i], kernel='linear',random_state=0, probability=True)
            clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
        choose = np.argmax(acc)
        allclf.append(SVC(C=param[choose], kernel='linear',random_state=0, probability=True).fit(X,y))
        file.write(str(item)+"SVMLN,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
        print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
        print("test_acc:", str(acc), ", test_f1:", str(f1))

        #RF
        print("Stacking-RF...")
        param = [20, 50, 100, 200]
        acc = np.zeros(len(param)) 
        sens = np.zeros(len(param)) 
        spec = np.zeros(len(param)) 
        mcc = np.zeros(len(param)) 
        roc = np.zeros(len(param)) 
        f1 = np.zeros(len(param))
        for i in range(0,len(param)):
            level0 = get_stacking()[:j+1]
            level1 = RandomForestClassifier(n_estimators=param[i], random_state=0)
            clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
        choose = np.argmax(acc)
        allclf.append(RandomForestClassifier(n_estimators=param[choose], random_state=0).fit(X,y))
        file.write(str(item)+"RF,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
        print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
        print("test_acc:", str(acc), ", test_f1:", str(f1))

        #E-Tree
        print("Stacking-ExTree...")
        param = [20, 50, 100, 200]
        acc = np.zeros(len(param)) 
        sens = np.zeros(len(param)) 
        spec = np.zeros(len(param)) 
        mcc = np.zeros(len(param)) 
        roc = np.zeros(len(param)) 
        f1 = np.zeros(len(param))
        for i in range(0,len(param)):
            level0 = get_stacking()[:j+1]
            level1 = ExtraTreesClassifier(n_estimators=param[i], random_state=0)
            clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
        choose = np.argmax(acc)
        allclf.append(ExtraTreesClassifier(n_estimators=param[choose], random_state=0).fit(X,y))
        file.write(str(item)+"ET,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
        print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
        print("test_acc:", str(acc), ", test_f1:", str(f1))

        #XGBoost
        print("Stacking-XGBoost...")
        param = [20, 50, 100, 200]
        acc = np.zeros(len(param)) 
        sens = np.zeros(len(param)) 
        spec = np.zeros(len(param)) 
        mcc = np.zeros(len(param)) 
        roc = np.zeros(len(param)) 
        f1 = np.zeros(len(param)) 
        for i in range(0,len(param)):
            level0 = get_stacking()[:j+1]
            level1 = XGBClassifier(n_estimators=param[i],learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=0)
            clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
        choose = np.argmax(acc)  
        allclf.append(XGBClassifier(n_estimators=param[choose],learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=0).fit(X,y))
        file.write(str(item)+"XGB,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
        print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
        print("test_acc:", str(acc), ", test_f1:", str(f1))

        #LightGBM
        print("Stacking-LightGBM...")
        param = [20, 50, 100, 200]
        acc = np.zeros(len(param)) 
        sens = np.zeros(len(param)) 
        spec = np.zeros(len(param)) 
        mcc = np.zeros(len(param)) 
        roc = np.zeros(len(param)) 
        f1 = np.zeros(len(param)) 
        for i in range(0,len(param)):
            level0 = get_stacking()[:j+1]
            level1 = LGBMClassifier(n_estimators=param[i],learning_rate=0.1, random_state=0)
            clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
        choose = np.argmax(acc)  
        allclf.append(LGBMClassifier(n_estimators=param[choose],learning_rate=0.1, random_state=0).fit(X,y))
        file.write(str(item)+"LGBM,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
        print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
        print("test_acc:", str(acc), ", test_f1:", str(f1))
        
        #MLP
        print("Stacking-MLP...")
        param = [20, 50, 100, 200]
        acc = np.zeros(len(param)) 
        sens = np.zeros(len(param)) 
        spec = np.zeros(len(param)) 
        mcc = np.zeros(len(param)) 
        roc = np.zeros(len(param)) 
        f1 = np.zeros(len(param))
        for i in range(0,len(param)):
            level0 = get_stacking()[:j+1]
            level1 = MLPClassifier(hidden_layer_sizes=(param[i],),random_state=0, max_iter = 10000)
            clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
        choose = np.argmax(acc)
        allclf.append(MLPClassifier(hidden_layer_sizes=(param[choose],),random_state=0, max_iter=10000).fit(X,y))
        file.write(str(item)+"MLP,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose])) 
        print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
        print("test_acc:", str(acc), ", test_f1:", str(f1))

        #NB
        print("Stacking-NB...")
        level0 = get_stacking()[:j+1] 
        level1 = GaussianNB()
        clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)
        acc, sens, spec, mcc, roc, f1 = test(clf,X,y,Xv,yv)
        allclf.append(clf)
        file.write(str(item)+"NB,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA"))
        print("val_acc:", str(acc), " ,val_f1:", str(f1))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
        print("test_acc:", str(acc), ", test_f1:", str(f1))

        #1NN
        print("Stacking-1NN...")
        level0 = get_stacking()[:j+1]
        level1 = KNeighborsClassifier(n_neighbors=1)
        clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)
        acc, sens, spec, mcc, roc, f1 = test(clf,X,y,Xv,yv)
        allclf.append(clf)
        file.write(str(item)+"1NN,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA"))
        print("val_acc:", str(acc), " ,val_f1:", str(f1))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
        print("test_acc:", str(acc), ", test_f1:", str(f1))

        #DT
        print("Stacking-DT...")
        level0 = get_stacking()[:j+1]
        level1 = DecisionTreeClassifier(random_state=0)
        clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)
        acc, sens, spec, mcc, roc, f1 = test(clf,X,y,Xv,yv)
        allclf.append(clf)
        file.write(str(item)+"DT,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA")) 
        print("val_acc:", str(acc), " ,val_f1:", str(f1))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
        print("test_acc:", str(acc), ", test_f1:", str(f1))
        
        #Logistic
        print("Stacking-LR...")
        param = [0.001,0.01,0.1,1,10,100]
        acc = np.zeros(len(param)) 
        sens = np.zeros(len(param)) 
        spec = np.zeros(len(param)) 
        mcc = np.zeros(len(param)) 
        roc = np.zeros(len(param)) 
        f1 = np.zeros(len(param))
        for i in range(0,len(param)):
            level0 = get_stacking()[:j+1]
            level1 = LogisticRegression(C=param[i], random_state=0, max_iter=10000)
            clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
        choose = np.argmax(acc)
        allclf.append(LogisticRegression(C=param[choose], random_state=0, max_iter=10000).fit(X,y))
        file.write(str(item)+"LR,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))   
        print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
        print("test_acc:", str(acc), ", test_f1:", str(f1))
        
        #PLS
        print("Stacking-PLS...")
        level0 = get_stacking()[:j+1]
        level1 = OneVsRestClassifier(PLS())
        clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=5, n_jobs=-1)
        acc, sens, spec, mcc, roc, f1 = test(clf,X,y,Xv,yv)
        allclf.append(clf)
        file.write(str(item)+"PLS,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA"))
        print("val_acc:", str(acc), " ,val_f1:", str(f1))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
        print("test_acc:", str(acc), ", test_f1:", str(f1))
    

    file.close()
