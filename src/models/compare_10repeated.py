import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import src.feature.build_features as bf
import src.utilities as utils
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearnex import patch_sklearn
from src.models.metrics import test
from src.models.PLS import PLS
from xgboost import XGBClassifier


def run(data_name, iname, df_ds, min_max):
    # try using https://github.com/intel/scikit-learn-intelex for accelerated implementations of algorithms 
    patch_sklearn(["SVC"]) # we patched only SVC here, since it tooks the longest.
    
  
    y_ds = df_ds['target'].astype('category').cat.codes
    Xo = df_ds['processed']
    yo = y_ds.to_numpy()


    for item in range(0, 10):
        print(data_name + ", " + iname , ", SEED:", item)

        X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=item, stratify=yo)
        X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)

        if iname == "POSMEAN":
            X_train_val = bf.extract(iname, X_train, min_max)
            X_val_val = bf.extract(iname, X_val, min_max)
            X_test_val = bf.extract(iname, X_test, min_max)
        elif iname == "POSW2V":
            _, X_train_val = bf.extract(iname, X_train, min_max)
            _, X_val_val = bf.extract(iname, X_val, min_max)
            _, X_test_val = bf.extract(iname, X_test, min_max)
        else:
            fe, X_train_val = bf.extract(iname, X_train, min_max)
            X_val_val = fe.transform(X_val)
            X_test_val = fe.transform(X_test)

       
        scaler = MaxAbsScaler()
        scaler.fit(X_train_val)
        X = scaler.transform(X_train_val)
        Xv = scaler.transform(X_val_val)
        Xt = scaler.transform(X_test_val)  

        file = open(config['output'] + str(item) +"_12classifier_"+iname+ "_" + str(min_max) + "_" + data_name + ".csv", "a")
        allclf = []

        
        #SVM
        print("SVM...")
        param = [1,2,4,8,16, 32]
        acc = np.zeros(len(param)) 
        sens = np.zeros(len(param)) 
        spec = np.zeros(len(param)) 
        mcc = np.zeros(len(param)) 
        roc = np.zeros(len(param)) 
        f1 = np.zeros(len(param))
        for i in range(0,len(param)):
            start_time = datetime.now()
            clf = SVC(C=param[i], random_state=0, probability=True)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
            elapsed = datetime.now() - start_time
            print(str(i) , ": took =", elapsed)
            print(acc[i], ", ", f1[i])
        choose = np.argmax(acc)
        allclf.append(SVC(C=param[choose], random_state=0, probability=True).fit(X,y))
        file.write(str(item)+"SVMRBF,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

        #LinearSVC
        print("Linear-SVM (sklearn.svm.LinearSVC)...")
        param = [1,2,4,8,16 ,32]
        acc = np.zeros(len(param)) 
        sens = np.zeros(len(param)) 
        spec = np.zeros(len(param)) 
        mcc = np.zeros(len(param)) 
        roc = np.zeros(len(param))
        f1 = np.zeros(len(param)) 
        
        for i in range(0,len(param)):
            start_time = datetime.now()
            clf =  SVC(C=param[i], kernel='linear',random_state=0, probability=True)
            #svm = LinearSVC(C=param[i],random_state=0, max_iter=20000)
            #clf = CalibratedClassifierCV(svm, cv=3) 
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
    
            elapsed = datetime.now() - start_time
            print(str(i) , ": took =", elapsed)
        choose = np.argmax(acc)
        allclf.append(SVC(C=param[i], kernel='linear',random_state=0, probability=True).fit(X,y))
        file.write(str(item)+"SVMLN,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")
        

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
            start_time = datetime.now()
            clf = RandomForestClassifier(n_estimators=param[i], random_state=0)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
            elapsed = datetime.now() - start_time
            print(str(i) , ": took =", elapsed)
        choose = np.argmax(acc)
        allclf.append(RandomForestClassifier(n_estimators=param[choose], random_state=0).fit(X,y))
        file.write(str(item)+"RF,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

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
            start_time = datetime.now()
            clf = ExtraTreesClassifier(n_estimators=param[i], random_state=0)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
            elapsed = datetime.now() - start_time
            print(str(i) , ": took =", elapsed)
        choose = np.argmax(acc)
        allclf.append(ExtraTreesClassifier(n_estimators=param[choose], random_state=0).fit(X,y))
        file.write(str(item)+"ET,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

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
            start_time = datetime.now()
            clf = XGBClassifier(n_estimators=param[i],learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=0)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
            elapsed = datetime.now() - start_time
            print(str(i) , ": took =", elapsed)
        choose = np.argmax(acc)  
        allclf.append(XGBClassifier(n_estimators=param[i],learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=0).fit(X,y))
        file.write(str(item)+"XGB,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

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
            start_time = datetime.now()
            clf = LGBMClassifier(n_estimators=param[i],learning_rate=0.1, random_state=0)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
            elapsed = datetime.now() - start_time
            print(str(i) , ": took =", elapsed)
        choose = np.argmax(acc)  
        allclf.append(LGBMClassifier(n_estimators=param[i],learning_rate=0.1, random_state=0).fit(X,y))
        file.write(str(item)+"LGBM,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))  
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

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
            start_time = datetime.now()
            clf = MLPClassifier(hidden_layer_sizes=(param[i],),random_state=0, max_iter = 10000)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
            elapsed = datetime.now() - start_time
            print(str(i) , ": took =", elapsed)
        choose = np.argmax(acc)
        allclf.append(MLPClassifier(hidden_layer_sizes=(param[choose],),random_state=0, max_iter=10000).fit(X,y))
        file.write(str(item)+"MLP,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose])) 
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

        #NB
        print("NB...")
        clf = GaussianNB()
        acc, sens, spec, mcc, roc, f1 = test(clf,X.toarray(),y,Xv.toarray(),yv)
        allclf.append(clf)
        file.write(str(item)+"NB,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA"))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

        #1NN
        print("1NN...")
        clf = KNeighborsClassifier(n_neighbors=1)
        acc, sens, spec, mcc, roc, f1 = test(clf,X,y,Xv,yv)
        allclf.append(clf)
        file.write(str(item)+"1NN,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA"))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

        #DT
        print("DT...")
        clf = DecisionTreeClassifier(random_state=0)
        acc, sens, spec, mcc, roc, f1 = test(clf,X,y,Xv,yv)
        allclf.append(clf)
        file.write(str(item)+"DT,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA")) 
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

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
            start_time = datetime.now()
            clf = LogisticRegression(C=param[i], random_state=0, max_iter=10000)
            acc[i], sens[i], spec[i], mcc[i], roc[i], f1[i] = test(clf,X,y,Xv,yv)
            elapsed = datetime.now() - start_time
            print(str(i) , ": took =", elapsed)
        choose = np.argmax(acc)
        allclf.append(LogisticRegression(C=param[choose], random_state=0, max_iter=10000).fit(X,y))
        file.write(str(item)+"LR,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(f1[choose])+","+str(param[choose]))   
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

        #PLS
        print("PLS...")
        clf = OneVsRestClassifier(PLS())
        acc, sens, spec, mcc, roc, f1 = test(clf,X.toarray(),y,Xv.toarray(),yv)
        allclf.append(clf)
        file.write(str(item)+"PLS,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+","+str("NA"))
        acc, sens, spec, mcc, roc, f1 = test(allclf[-1], np.vstack((X.toarray(),Xv.toarray())), np.hstack((y,yv)), Xt.toarray(), yt)
        file.write(","+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str(f1)+"\n")

        file.close()
    return 


if __name__ == "__main__":
    config = utils.read_config()
    if len(sys.argv) != 4:
        print("*Error: incorrect number of arguments.")
        print("*Usage:[dataset name]", config['feature']['build_method'], "[min,max]")
        sys.exit(1)

    elif sys.argv[2] in config['feature']['build_method'] and sys.argv[1] in config['data']['name']: 
        data_name = sys.argv[1]
        text_rep = sys.argv[2]
        min_max = sys.argv[3]

        print("Loading and converting from csv...")
        if data_name == 'kt':
            df_ds = pd.read_csv(config['data']['processed_kt'])
        elif data_name == 'ws':
            df_ds = pd.read_csv(config['data']['processed_ws'])
        elif data_name == 'tt':
            df_ds = pd.read_csv(config['data']['processed_tt'])
        elif data_name == 'wn':
            Xo, yo = joblib.load(config['data']['processed_wn'])
            df_ds = pd.DataFrame(list(zip(Xo, yo)), columns=['processed', 'target'])
        else:
            sys.exit(1)
        print("*Modeling ", text_rep, "representation(s) ", "ngram = ", min_max, "for: ", data_name)
        run(data_name, text_rep, df_ds, tuple(map(int, min_max.split(','))))
    else:
        print("*Error: incorrect argument name or dataset name.")
        sys.exit(1)
    print('*Program terminate successfully!')
