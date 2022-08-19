from pathlib import Path

import numpy as np
import src.utilities as utils
import torch
from lightgbm import LGBMClassifier
from numpy import mean, std
from sklearn.datasets import load_svmlight_file, make_classification
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.models.camerbert import Camembert
from src.models.PLS import PLS
from xgboost import XGBClassifier

configs = utils.read_config()
root = utils.get_project_root()

######################################################################
data_path = str(Path.joinpath(root, configs['data']['wangcha_to']))
file = open(configs['output_scratch'] +"stacking_ensemble_to.csv", "a")
num_class = 2
######################################################################

def test(y_test, p, pr):
    ACC = accuracy_score(y_test, p)
    SENS = precision_score(y_test, p, average='macro')
    SPEC = recall_score(y_test,p, average='macro')
    MCC = matthews_corrcoef(y_test,p)
    if num_class > 2:
        AUC = roc_auc_score(y_test, pr, multi_class='ovo',average='macro')
    else:
        AUC = roc_auc_score(y_test,pr[:,1]) # for binary classification problem
    F1 = 2*SENS*SPEC/(SENS+SPEC)
    return ACC, SENS, SPEC, MCC, AUC, F1

def get_data(idx):
    data = load_svmlight_file(data_path + "\\" + "traindata_"+str(idx)+".scl", zero_based=False)
    data1 = load_svmlight_file(data_path + "\\" + "testdata_"+str(idx)+".scl", zero_based=False)
    return data[0].toarray(), data[1], data1[0].toarray(), data1[1]


def get_stacking():
	# define the base models
    level0 = list()
    #level0.append(('mlp', MLPClassifier(hidden_layer_sizes=50, random_state=0, max_iter = 10000)))
    #level0.append(('knn', KNeighborsClassifier()))
    level0.append(('rf', RandomForestClassifier( random_state=0)))
    #level0.append(('svm', SVC(random_state=0, probability=True)))
    # level0.append(('mnb', GaussianNB()))
    # level0.append(('lgbm', LGBMClassifier()))
    # level0.append(('et', ExtraTreesClassifier()))
    # level0.append(('xgb', XGBClassifier() ))
    level0.append(('pls', OneVsRestClassifier(PLS())))
    
    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=10, n_jobs=-1)
    return model
 
def get_models():
	models = dict()
	# models['lr'] = LogisticRegression()
	# models['knn'] = KNeighborsClassifier()
	# models['rf'] = RandomForestClassifier()
	#models['svm'] = SVC(random_state=0, probability=True)
	# models['bayes'] = GaussianNB()
	models['blending'] = get_stacking()
	return models

def evaluate_model(item, model, X, y):
    val_f1 = 0
    test_f1 = 0
    
    idx = int(round(len(ya)*0.75))
    X = Xa[0:idx, :]
    y = ya[0:idx]
    Xv = Xa[idx:-1, :]
    yv = ya[idx:-1]

    model.fit(X, y)
    p = model.predict(Xv)
    pr = model.predict_proba(Xv)
    acc, pre, rec, mcc, auc, f1 = test(yv, p, pr)
    val_f1 = f1
    file.write(str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1))
    
    model.fit(np.vstack((X,Xv)), np.hstack((y,yv)))
    p = model.predict(Xt)
    pr = model.predict_proba(Xt)
    acc, pre, rec, mcc, auc, f1 = test(yt, p, pr)
    test_f1 = f1
    file.write("," + str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1) +"\n") 
    return val_f1, test_f1
        
models = get_models()
val_f1_all = 0
test_f1_all = 0
for item in range(0, 10):
    for name, model in models.items():
        Xa, ya, Xt, yt = get_data(item)
        val_f1, test_f1 = evaluate_model(item, model, Xa, ya)
        val_f1_all += val_f1
        test_f1_all += test_f1
print("val_f1:", val_f1_all/10, "test_f1:", test_f1_all/10)
file.close()

