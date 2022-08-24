import sys
from pathlib import Path

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import src.utilities as utils
from sklearn.preprocessing import MinMaxScaler

SEED = [i for i in range(1,10)]


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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
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
out_file_name = 'ws_tuned_level0.csv' 
num_class = 4
##################################################################

patch_sklearn()

# def feature_selection(X, y):
#     from sklearn.feature_selection import SelectKBest, mutual_info_classif
#     selector = SelectKBest(mutual_info_classif, k=45)
#     X_new = selector.fit_transform(X, y)
#     return X_new

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

iname = "WANGCHAN-STACKING"
for item in SEED:
    print("SEED:", item)

#for j in range(len(get_stacking())):

Xa, ya, Xt, yt = get_data(item)
print(Xa.shape)

idx = int(round(len(ya)*0.75))
X = Xa[0:idx, :]
y = ya[0:idx]
Xv = Xa[idx:-1, :]
yv = ya[idx:-1]

allclf = []
file = open("12classifier_"+iname+"_res_" + out_file_name, "a")
print(out_file_name)


print("Stacking-SVM...")
#SVM
param = [1,2,4,8,16,32]
acc = np.zeros(len(param)) 
sens = np.zeros(len(param)) 
spec = np.zeros(len(param)) 
mcc = np.zeros(len(param)) 
roc = np.zeros(len(param)) 
f1 = np.zeros(len(param))

grid_param = {
'svm__C': [1,2,4,8,16,32], 'svm__gamma': [1,0.1,0.01,0.001],
'mlp__hidden_layer_sizes': [10, 20, 50, 100, 200],
'xgb__n_estimators': [10, 20, 50, 100, 200],
'rf__n_estimators': [20, 50, 100, 200, 400],
'et__n_estimators': [20, 50, 100, 200, 400],
'lgbm__n_estimators': [20, 50, 100, 200],
}

level0 = get_stacking()
svc = SVC(random_state=0, probability=True)
clfs = StackingClassifier(estimators=level0, final_estimator=svc, cv=5, n_jobs=-1)
level1 =  GridSearchCV(estimator=clfs, param_grid=grid_param, cv=5, n_jobs=-1)
level1.fit(X, y)
best_clf = level1.best_estimator_
acc, sens, spec, mcc, roc, f1 = test(best_clf,X,y,Xv,yv)
file.write(str(item)+"SVMRBF,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str(param))  
print("val_acc:", acc, " ,val_f1:", str(f1))
acc, sens, spec, mcc, roc, f1 = test(clf, np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
print("val_acc:", str(acc), ", test_f1:", str(f1))

#LinearSVC
print("Stacking-LinearSVC...")
level0 = get_stacking()
lsvc = SVC(C=param[i], kernel='linear',random_state=0, probability=True)
clfs = StackingClassifier(estimators=level0, final_estimator=lsvc, cv=5, n_jobs=-1)
level1 =  GridSearchCV(estimator=clfs, param_grid=grid_param, cv=5, n_jobs=-1)
level1.fit(X, y)
best_clf = level1.best_estimator_
acc, sens, spec, mcc, roc, f1 = test(best_clf,X,y,Xv,yv)
file.write(str(item)+"SVMLN,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str(param))  
print("val_acc:", acc, " ,val_f1:", str(f1))
acc, sens, spec, mcc, roc, f1 = test(clf, np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
print("val_acc:", str(acc), ", test_f1:", str(f1))

#RF
print("Stacking-RF...")
level0 = get_stacking()
rf = RandomForestClassifier(random_state=0)
clfs = StackingClassifier(estimators=level0, final_estimator=rf, cv=5, n_jobs=-1)
level1 =  GridSearchCV(estimator=clfs, param_grid=grid_param, cv=5, n_jobs=-1)
level1.fit(X, y)
best_clf = level1.best_estimator_
acc, sens, spec, mcc, roc, f1 = test(best_clf,X,y,Xv,yv)
file.write(str(item)+"RF,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str(param))  
print("val_acc:", acc, " ,val_f1:", str(f1))
acc, sens, spec, mcc, roc, f1 = test(clf, np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
print("val_acc:", str(acc), ", test_f1:", str(f1))

# #E-Tree
print("Stacking-ExTree...")
level0 = get_stacking()
et = ExtraTreesClassifier(random_state=0)
clfs = StackingClassifier(estimators=level0, final_estimator=et, cv=5, n_jobs=-1)
level1 =  GridSearchCV(estimator=clfs, param_grid=grid_param, cv=5, n_jobs=-1)
level1.fit(X, y)
best_clf = level1.best_estimator_
acc, sens, spec, mcc, roc, f1 = test(best_clf,X,y,Xv,yv)
file.write(str(item)+"ET,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str(param))  
print("val_acc:", acc, " ,val_f1:", str(f1))
acc, sens, spec, mcc, roc, f1 = test(clf, np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
print("val_acc:", str(acc), ", test_f1:", str(f1))

# #XGBoost
print("Stacking-XGBoost...")
level0 = get_stacking()
xgb = XGBClassifier(learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=0)
clfs = StackingClassifier(estimators=level0, final_estimator=xgb, cv=5, n_jobs=-1)
level1 =  GridSearchCV(estimator=clfs, param_grid=grid_param, cv=5, n_jobs=-1)
level1.fit(X, y)
best_clf = level1.best_estimator_
acc, sens, spec, mcc, roc, f1 = test(best_clf,X,y,Xv,yv)
file.write(str(item)+"XGB,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str(param))  
print("val_acc:", acc, " ,val_f1:", str(f1))
acc, sens, spec, mcc, roc, f1 = test(clf, np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
print("val_acc:", str(acc), ", test_f1:", str(f1))

# #LightGBM
print("Stacking-LightGBM...")
level0 = get_stacking()
lgbm = LGBMClassifier(learning_rate=0.1, random_state=0)
clfs = StackingClassifier(estimators=level0, final_estimator=lgbm, cv=5, n_jobs=-1)
level1 =  GridSearchCV(estimator=clfs, param_grid=grid_param, cv=5, n_jobs=-1)
level1.fit(X, y)
best_clf = level1.best_estimator_
acc, sens, spec, mcc, roc, f1 = test(best_clf,X,y,Xv,yv)
file.write(str(item)+"LGBM,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str(param))  
print("val_acc:", acc, " ,val_f1:", str(f1))
acc, sens, spec, mcc, roc, f1 = test(clf, np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
print("val_acc:", str(acc), ", test_f1:", str(f1))

# #MLP
print("Stacking-MLP...")
level0 = get_stacking()
mlp = MLPClassifier(random_state=0, max_iter=10000)
clfs = StackingClassifier(estimators=level0, final_estimator=mlp, cv=5, n_jobs=-1)
level1 =  GridSearchCV(estimator=clfs, param_grid=grid_param, cv=5, n_jobs=-1)
level1.fit(X, y)
best_clf = level1.best_estimator_
acc, sens, spec, mcc, roc, f1 = test(best_clf,X,y,Xv,yv)
file.write(str(item)+"MLP,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str(param))  
print("val_acc:", acc, " ,val_f1:", str(f1))
acc, sens, spec, mcc, roc, f1 = test(clf, np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
print("val_acc:", str(acc), ", test_f1:", str(f1))

#NB
print("Stacking-NB...")
level0 = get_stacking()
nb = GaussianNB()
clfs = StackingClassifier(estimators=level0, final_estimator=nb, cv=5, n_jobs=-1)
level1 =  GridSearchCV(estimator=clfs, param_grid=grid_param, cv=5, n_jobs=-1)
level1.fit(X, y)
best_clf = level1.best_estimator_
file.write(str(item)+"NB,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str(param))  
print("val_acc:", acc, " ,val_f1:", str(f1))
acc, sens, spec, mcc, roc, f1 = test(clf, np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
print("val_acc:", str(acc), ", test_f1:", str(f1))

# #1NN
print("Stacking-1NN...")
level0 = get_stacking()
knn = KNeighborsClassifier(n_neighbors=1)
clfs = StackingClassifier(estimators=level0, final_estimator=knn, cv=5, n_jobs=-1)
level1 =  GridSearchCV(estimator=clfs, param_grid=grid_param, cv=5, n_jobs=-1)
level1.fit(X, y)
best_clf = level1.best_estimator_
acc, sens, spec, mcc, roc, f1 = test(best_clf,X,y,Xv,yv)
file.write(str(item)+"1NN,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str(param))  
print("val_acc:", acc, " ,val_f1:", str(f1))
acc, sens, spec, mcc, roc, f1 = test(clf, np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
print("val_acc:", str(acc), ", test_f1:", str(f1))

# #DT
print("Stacking-DT...")
level0 = get_stacking()
dt = DecisionTreeClassifier(random_state=0)
clfs = StackingClassifier(estimators=level0, final_estimator=dt, cv=5, n_jobs=-1)
level1 =  GridSearchCV(estimator=clfs, param_grid=grid_param, cv=5, n_jobs=-1)
level1.fit(X, y)
best_clf = level1.best_estimator_
acc, sens, spec, mcc, roc, f1 = test(best_clf,X,y,Xv,yv)
file.write(str(item)+"DT,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str(param))  
print("val_acc:", acc, " ,val_f1:", str(f1))
acc, sens, spec, mcc, roc, f1 = test(clf, np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
print("val_acc:", str(acc), ", test_f1:", str(f1))

# #Logistic
print("Stacking-LR...")
level0 = get_stacking()
lr = LogisticRegression(random_state=0, max_iter=10000)
clfs = StackingClassifier(estimators=level0, final_estimator=lr, cv=5, n_jobs=-1)
level1 =  GridSearchCV(estimator=clfs, param_grid=grid_param, cv=5, n_jobs=-1)
level1.fit(X, y)
best_clf = level1.best_estimator_
acc, sens, spec, mcc, roc, f1 = test(best_clf,X,y,Xv,yv)
file.write(str(item)+"LR,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str(param))  
print("val_acc:", acc, " ,val_f1:", str(f1))
acc, sens, spec, mcc, roc, f1 = test(clf, np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
print("val_acc:", str(acc), ", test_f1:", str(f1))

# #PLS
print("Stacking-PLS...")
level0 = get_stacking()
pls = OneVsRestClassifier(PLS())
clfs = StackingClassifier(estimators=level0, final_estimator=pls, cv=5, n_jobs=-1)
level1 =  GridSearchCV(estimator=clfs, param_grid=grid_param, cv=5, n_jobs=-1)
level1.fit(X, y)
best_clf = level1.best_estimator_
acc, sens, spec, mcc, roc, f1 = test(best_clf,X,y,Xv,yv)
file.write(str(item)+"PLS,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str(param))  
print("val_acc:", acc, " ,val_f1:", str(f1))
acc, sens, spec, mcc, roc, f1 = test(clf, np.vstack((X,Xv)), np.hstack((y,yv)), Xt, yt)
file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
print("val_acc:", str(acc), ", test_f1:", str(f1))

file.close()
