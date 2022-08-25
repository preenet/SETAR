import sys
from pathlib import Path

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import src.utilities as utils
from pyexpat.errors import XML_ERROR_INVALID_TOKEN
from sklearn.preprocessing import MinMaxScaler

SEED = [i for i in range(0,10)]


from lightgbm import LGBMClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearnex import patch_sklearn
from src.models.PLS import PLS
from xgboost import XGBClassifier

configs = utils.read_config()
root = utils.get_project_root()

##################################################################
data_path = str(Path.joinpath(root, configs['data']['wangcha_ws_feature']))
out_file_name = 'ws_hstack.csv' 
num_class = 4
##################################################################

patch_sklearn()

# def feature_selection(X, y):
#     from sklearn.feature_selection import SelectKBest, mutual_info_classif
#     selector = SelectKBest(mutual_info_classif, k=45)
#     X_new = selector.fit_transform(X, y)
#     return X_new

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

def get_data_fe(idx):
    fe = ['WANCHAN', 'BOW1' , 'TF1', 'W2V']
    #for j in enumerate(fe):
    tr1 = load_svmlight_file(data_path + "\\" + "traindata_" + str(fe[0]) +"_"+ str(idx)+ ".scl", zero_based=False)
    tr2 = load_svmlight_file(data_path + "\\" + "traindata_" + str(fe[1]) +"_"+ str(idx)+ ".scl", zero_based=False)
    tr3 = load_svmlight_file(data_path + "\\" + "traindata_" + str(fe[2]) +"_"+ str(idx)+ ".scl", zero_based=False)
    tr4 = load_svmlight_file(data_path + "\\" + "traindata_" + str(fe[3]) +"_"+ str(idx)+ ".scl", zero_based=False)

    print(tr1[0].toarray().shape, tr2[0].toarray().shape, tr3[0].toarray().shape, tr4[0].toarray().shape)
    data = np.hstack(( tr1[0].toarray(), tr2[0].toarray(), tr3[0].toarray(), tr4[0].toarray()))

    t1 = load_svmlight_file(data_path + "\\" + "testdata_" + str(fe[0]) +"_"+ str(idx)+ ".scl", zero_based=False)
    t2 = load_svmlight_file(data_path + "\\" + "testdata_" + str(fe[1]) +"_"+ str(idx)+ ".scl", zero_based=False)
    t3 = load_svmlight_file(data_path + "\\" + "testdata_" + str(fe[2]) +"_"+ str(idx)+ ".scl", zero_based=False)
    t4 = load_svmlight_file(data_path + "\\" + "testdata_" + str(fe[3]) +"_"+ str(idx)+ ".scl", zero_based=False)
    print(t1[0].toarray().shape, t2[0].toarray().shape, t3[0].toarray().shape, t4[0].toarray().shape)
    data1 = np.hstack((t1[0].toarray(), t2[0].toarray(), t3[0].toarray(), t4[0].toarray()))
    return data, tr4[1], data1, t4[1]

iname = "WANGCHAN-FeatureStacked"

for item in SEED:
    print("SEED:", item)
    Xa, ya, Xt, yt = get_data_fe(item)
    print(Xa.shape)

    scaler = MaxAbsScaler()
    scaler.fit(Xa)
    scaler.transform(Xa)
    
    X, X_tmp, y, y_tmp = train_test_split(Xa, ya, test_size=0.4, random_state=item, stratify=ya)
    Xv, Xt, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)
    
    allclf = []
    file = open("12classifier_"+iname+"_res_" + out_file_name, "a")
    #SVM
    print("Stacking-SVM...")
    param = [1,2,4,8,16, 32]
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
    print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
    print("test_acc:", str(acc), ", test_f1:", str(f1))

    #LinearSVC
    print("Linear-SVM..")
    param = [1,2,4,8,16 ,32]
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
    print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
    print("test_acc:", str(acc), ", test_f1:", str(f1))
    

    #RF
    print("RF...")
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
    print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
    print("test_acc:", str(acc), ", test_f1:", str(f1))

    #E-Tree
    print("E-tree...")
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
    print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
    print("test_acc:", str(acc), ", test_f1:", str(f1))

    #XGBoost
    print("XGBoost...")
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
    print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
    print("test_acc:", str(acc), ", test_f1:", str(f1))

    #LightGBM
    print("LightGBM...")
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
    print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
    print("test_acc:", str(acc), ", test_f1:", str(f1))

    #MLP
    print("MLP...")
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
    print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
    print("test_acc:", str(acc), ", test_f1:", str(f1))

    #NB
    print("NB...")
    clf = GaussianNB()
    acc, sens, spec, mcc, roc, f1 = test(clf,X,y,yv)
    allclf.append(clf)
    file.write(str(item)+"NB,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA"))
    print("val_acc:", str(acc), " ,val_f1:", str(f1))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
    print("test_acc:", str(acc), ", test_f1:", str(f1))

    #1NN
    print("1NN...")
    clf = KNeighborsClassifier(n_neighbors=1)
    acc, sens, spec, mcc, roc, f1 = test(clf,X,y,Xv,yv)
    allclf.append(clf)
    file.write(str(item)+"1NN,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA"))
    print("val_acc:", str(acc), " ,val_f1:", str(f1))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
    print("test_acc:", str(acc), ", test_f1:", str(f1))

    #DT
    print("DT...")
    clf = DecisionTreeClassifier(random_state=0)
    acc, sens, spec, mcc, roc, f1 = test(clf,X,y,Xv,yv)
    allclf.append(clf)
    file.write(str(item)+"DT,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA")) 
    print("val_acc:", str(acc), " ,val_f1:", str(f1))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
    print("test_acc:", str(acc), ", test_f1:", str(f1))

    #Logistic
    print("LR...")
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
    print("val_acc:", acc[choose], " ,val_f1:", str(f1[choose]))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
    print("test_acc:", str(acc), ", test_f1:", str(f1))

    #PLS
    print("PLS...")
    clf = OneVsRestClassifier(PLS())
    acc, sens, spec, mcc, roc, f1 = test(clf,X,y,Xv,yv)
    allclf.append(clf)
    file.write(str(item)+"PLS,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA"))
    print("val_acc:", str(acc), " ,val_f1:", str(f1))
    acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
    file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
    print("test_acc:", str(acc), ", test_f1:", str(f1))

    file.close()
