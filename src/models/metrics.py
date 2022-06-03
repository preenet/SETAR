from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from keras import backend as K

__all__ = ['CV', 'test', 'f1_m']

def CV(model, X_train, y_train, X_valid, y_valid):
    scores = (cross_val_score(model, X_train, y_train, cv = 5).mean())
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    acc_sc = accuracy_score(y_valid, y_pred)
    pre_sc = precision_score(y_valid, y_pred, average='weighted')
    rec_sc = recall_score(y_valid, y_pred, average='weighted')
    f1_sc = f1_score(y_valid, y_pred, average='weighted')
    return acc_sc, pre_sc, rec_sc, f1_sc, scores


        
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
    F1 = 2*SENS*SPEC/(SENS+SPEC)
    return ACC, SENS, SPEC, MCC, AUC, F1


