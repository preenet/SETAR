## base-line with LR 5 fold CV
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import  accuracy_score
from sklearn.preprocessing import MaxAbsScaler
import src.utilities as utils
import src.feature.build_features as bf
from src.models.metrics import CV


config = utils.read_config()

df_ds = pd.read_csv(config['data']['processed_ws'])
y_ds = df_ds['target'].astype('category').cat.codes

Xo = df_ds['processed']
yo = y_ds.to_numpy()
text_reps = ['W2V']
file = open(config['output_scratch']+"KFoldLR.csv", "a")
file.write("valid_acc, precision,recall, f1_score, mean_cv, test_acc" + "\n") 

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
        
        model = LogisticRegression(C=2., penalty="l2", solver="liblinear", dual=False, multi_class="ovr")

        acc_sc, pre_sc, rec_sc, f1_sc, scores = CV(model, X, y, Xv, yv)
        yhat = model.predict(Xt)
        test_acc_sc = accuracy_score(yt, yhat)
        print(text_rep +':SEED= ', i, 'val= ', scores , ' test= %.4f' % (test_acc_sc * 100))
        file.write(str(acc_sc)+","+str(pre_sc)+","+str(rec_sc)+","+str(f1_sc)+","+str(scores)+","+str(test_acc_sc)+"\n") 
file.close()
    
    