"""
use SETAR-KT to predict testing sets from other datasets.
"""

from tkinter import Y

import joblib
import numpy as np
import src.utilities as utils
from sklearn.model_selection import train_test_split


def get_data():    
    configs = utils.read_config()

    x_tt, y_tt = joblib.load(configs['data']['kaggle_tt'])
    data = joblib.load(configs['data']['kaggle_to'])
    x_tx = data[0]
    y_tx = data[1]
    x_ws, y_ws = joblib.load(configs['data']['kaggle_ws'])
    x_kt, y_kt = joblib.load(configs['data']['kaggle_kt'])
    return x_tt, y_tt, x_tx, y_tx, x_ws, y_ws, x_kt, y_kt


if __name__ == "__main__":
    configs = utils.read_config()
    root = utils.get_project_root()
    
    print("SETAR-KT prediction powers:")
    setar_kt = joblib.load(configs['models'] +'SETAR-KT.model')
    
    print("Getting independent test for all datasets.")
    x_tt, y_tt, x_tx, y_tx, x_ws, y_ws, x_kt, y_kt  = get_data()

    #idx = [i for i, e in enumerate(y_tt) if e == 1]
 
    #y_tt = np.delete(y_tt, idx)
    #x_tt = np.delete(x_tt, idx)
    
    X_train, X_test, y_train, y_test = train_test_split(x_kt, y_kt, test_size=0.2, random_state=0, stratify=y_tt)
    # setar_kt.fit(X_train, y_train)
    # res = setar_kt.score(X_test)
    # print(res)
    
    print(setar_kt)
    
