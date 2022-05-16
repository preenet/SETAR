from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import MaxAbsScaler
import src.utilities as utils
import src.feature.build_features as bf
import pandas as pd
import warnings

config = utils.read_config()

df_ds = pd.read_csv(config['data']['processed_ws'])
y_ds = df_ds['target'].astype('category').cat.codes

Xo = df_ds['processed']
yo = y_ds.to_numpy()
text_reps = ['POSTFIDF']
file = open(config['output_scratch']+"KFoldLR.csv", "a")  

for text_rep in text_reps:
    file.write(text_rep+"\n")
    for i in range(0, 10):
        X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=i, stratify=yo)
        X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=i, stratify=y_tmp)

        fe, X_train_val = bf.extract(text_rep, X_train, (1,1))
        X_val_val = fe.transform(X_val)
        X_test_val = fe.transform(X_test)

        scaler = MaxAbsScaler()
        scaler.fit(X_train_val)
        X = scaler.transform(X_train_val)
        Xv = scaler.transform(X_val_val)
        Xt = scaler.transform(X_test_val)   

        def CV(model, X_train, y_train, X_valid, y_valid):
            scores = (cross_val_score(model, X_train, y_train, cv = 5).mean())
            model = model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            acc_sc = accuracy_score(y_valid, y_pred)
            pre_sc = precision_score(y_valid, y_pred, average='weighted')
            rec_sc = recall_score(y_valid, y_pred, average='weighted')
            f1_sc = f1_score(y_valid, y_pred, average='weighted')
            return acc_sc, pre_sc, rec_sc, f1_sc, scores
        
        model = LogisticRegression(C=2., penalty="l2", solver="liblinear", dual=False, multi_class="ovr")

        acc_sc, pre_sc, rec_sc, f1_sc, scores = CV(model, X, y, Xv, yv)
        yhat = model.predict(Xt)
        test_acc_sc = accuracy_score(yt, yhat)
        print(text_rep +':SEED= ', i, 'val= ', scores , ' test= %.4f' % (test_acc_sc * 100))
        file.write(str(acc_sc)+","+str(pre_sc)+","+str(rec_sc)+","+str(f1_sc)+","+str(scores)+","+str(test_acc_sc)+"\n") 
file.close()
    
    