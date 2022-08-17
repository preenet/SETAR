from pathlib import Path

import numpy as np
import src.utilities as utils
import torch
from lightgbm import LGBMClassifier
from matplotlib import pyplot
# try sklearn library
# compare ensemble to each baseline classifier
from numpy import mean, std
from sklearn.datasets import load_svmlight_file, make_classification
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.models.camerbert import Camembert
from xgboost import XGBClassifier

configs = utils.read_config()
root = utils.get_project_root()

data_path = str(Path.joinpath(root, configs['data']['wangcha_tt']))

def test(y_test, p, pr):
    ACC = accuracy_score(y_test, p)
    SENS = precision_score(y_test, p, average='macro')
    SPEC = recall_score(y_test,p, average='macro')
    MCC = matthews_corrcoef(y_test,p)
    AUC = roc_auc_score(y_test, pr, multi_class='ovo',average='macro')
    #AUC = roc_auc_score(test_y,pr[:,1]) # for binary classification problem
    F1 = 2*SENS*SPEC/(SENS+SPEC)
    return ACC, SENS, SPEC, MCC, AUC, F1

def get_data(idx):
    data = load_svmlight_file(data_path + "\\" + "traindata_"+str(idx)+".scl", zero_based=False)
    data1 = load_svmlight_file(data_path + "\\" + "testdata_"+str(idx)+".scl", zero_based=False)
    return data[0].toarray(), data[1], data1[0].toarray(), data1[1]

# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('mlp', MLPClassifier()))
	level0.append(('knn', KNeighborsClassifier()))
	level0.append(('rf', RandomForestClassifier()))
	level0.append(('svm', SVC(probability=True)))
	level0.append(('bayes', GaussianNB()))
	# define meta learner model
	level1 = LogisticRegression()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=2)
	return model
 
# get a list of models to evaluate
def get_models():
	models = dict()
	# models['lr'] = LogisticRegression()
	#models['knn'] = KNeighborsClassifier()
	# models['rf'] = RandomForestClassifier()
	# models['svm'] = SVC(probability=True)
	# models['bayes'] = GaussianNB()
	models['blending'] = get_stacking()
	return models
Xa, ya, Xt, yt = get_data(0)
def evaluate_model(model, X, y):
    
    idx = int(round(len(ya)*0.75))
    X = Xa[0:idx, :]
    y = ya[0:idx]
    Xv = Xa[idx:-1, :]
    yv = ya[idx:-1]
    
    for item in range(0, 10):
        file = open(configs['output_scratch'] +"stacking_ensemble_tt.csv", "a") 
        

        model.fit(X, y)
        p = model.predict(Xv)
        pr = model.predict_proba(Xv)
        acc, pre, rec, mcc, auc, f1 = test(yv, p, pr)
        file.write(str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1))
        
        model.fit(np.vstack((X,Xv)), np.hstack((y,yv)))
        p = model.predict(Xt)
        pr = model.predict_proba(Xt)
        acc, pre, rec, mcc, auc, f1 = test(yt, p, pr)
        file.write("," + str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1) +"\n") 
        file.close()

models = get_models()
for name, model in models.items():
	evaluate_model(model, Xa, ya)

