from pathlib import Path

import joblib
import numpy as np
import src.utilities as utils
import torch
from sklearn.metrics import matthews_corrcoef  # average == 'macro'.
from sklearn.metrics import \
    roc_auc_score  # multiclas 'ovo' average == 'macro'.
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from src.models.deep.camerbert import Camembert
from transformers import CamembertTokenizer, RobertaModel

configs = utils.read_config()
root = utils.get_project_root()

######################################################################
model_path = str(Path.joinpath(root, configs['wangchan_models']['ws']))
num_class = 4
#Xo, yo = joblib.load(Path.joinpath(root, configs['data']['kaggle_ws']))

data = joblib.load(Path.joinpath(root, configs['data']['kaggle_ws']))
Xo = data[0]
yo = data[1]
######################################################################



for item in range(0, 10):
    X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=item, stratify=yo)
    X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)

    train_wisesight = X_train
    validation_wisesight = X_val
    test_wisesight = X_test
        
    wangchan_tokenizer = CamembertTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased", use_fast=True)

    # Model parameter
    MAX_LEN = 256

    # Tokenized    
    tr_input_ids, tr_token_type_ids, tr_attention_mask = [],[],[]
    for seq in train_wisesight:
        encoding = wangchan_tokenizer.encode_plus(
            seq,
            add_special_tokens=True, 
            max_length=MAX_LEN, 
            padding='max_length', 
            return_attention_mask=True, 
            return_token_type_ids=True,
            truncation=True,
        )
        tr_input_ids.append(encoding['input_ids'])
        tr_token_type_ids.append(encoding['token_type_ids'])
        tr_attention_mask.append(encoding['attention_mask'])

    val_input_ids, val_token_type_ids, val_attention_mask = [],[],[]

    for seq in validation_wisesight:
        encoding = wangchan_tokenizer.encode_plus(
            seq,
            add_special_tokens=True, 
            max_length=MAX_LEN, 
            padding='max_length', 
            return_attention_mask=True, 
            return_token_type_ids=True,
            truncation=True,
        )
        val_input_ids.append(encoding['input_ids'])
        val_token_type_ids.append(encoding['token_type_ids'])
        val_attention_mask.append(encoding['attention_mask'])

    ts_input_ids, ts_token_type_ids, ts_attention_mask = [],[],[]

    for seq in test_wisesight:
        encoding = wangchan_tokenizer.encode_plus(
            seq,
            add_special_tokens=True, 
            max_length=MAX_LEN, 
            padding='max_length', 
            return_attention_mask=True, 
            return_token_type_ids=True,
            truncation=True,
        )
        ts_input_ids.append(encoding['input_ids'])
        ts_token_type_ids.append(encoding['token_type_ids'])
        ts_attention_mask.append(encoding['attention_mask'])
        
    model = Camembert(num_class)    
    model_file = "./model_"+str(item)+".pt"
    model.load_state_dict(torch.load(model_path + model_file))
    model.to('cuda')
    #y, yv, yt = train_wisesight['category'].values, validation_wisesight['category'].values, test_wisesight['category'].values

    X1 = torch.from_numpy(np.array(tr_input_ids)).long()
    X2 = torch.from_numpy(np.array(tr_attention_mask)).long()
    y = torch.from_numpy(y).float()

    Xv1 = torch.from_numpy(np.array(val_input_ids)).long()
    Xv2 = torch.from_numpy(np.array(val_attention_mask)).long()
    yv = torch.from_numpy(yv).float()


    Xt1 = torch.from_numpy(np.array(ts_input_ids)).long()
    Xt2 = torch.from_numpy(np.array(ts_attention_mask)).long()
    yt = torch.from_numpy(yt).float()


    from torch.utils.data import DataLoader, TensorDataset
    train_dl = DataLoader(TensorDataset(X1, X2, y), batch_size=32, shuffle=False)
    val_dl = DataLoader(TensorDataset(Xv1, Xv2, yv), batch_size=32, shuffle=False)
    test_dl = DataLoader(TensorDataset(Xt1, Xt2, yt), batch_size=32, shuffle=False)

    for i, (input1, input2, targets) in enumerate(train_dl):
        input1_gpu = input1.to('cuda', dtype = torch.long)
        input2_gpu = input2.to('cuda', dtype = torch.long)
        targets_gpu = targets.to('cuda', dtype = torch.long)
        yp = model.extract(input1_gpu, input2_gpu).cpu().detach().numpy() 
        if i == 0:
            X = yp
        else:
            X = np.vstack((X,yp))
            
    for i, (input1, input2, targets) in enumerate(val_dl):
        input1_gpu = input1.to('cuda', dtype = torch.long)
        input2_gpu = input2.to('cuda', dtype = torch.long)
        targets_gpu = targets.to('cuda', dtype = torch.long)
        yp = model.extract(input1_gpu, input2_gpu).cpu().detach().numpy() 
        if i == 0:
            Xv = yp
        else:
            Xv = np.vstack((Xv,yp))

    for i, (input1, input2, targets) in enumerate(test_dl):
        input1_gpu = input1.to('cuda', dtype = torch.long)
        input2_gpu = input2.to('cuda', dtype = torch.long)
        targets_gpu = targets.to('cuda', dtype = torch.long)
        yp = model.extract(input1_gpu, input2_gpu).cpu().detach().numpy() 
        if i == 0:
            Xt = yp
        else:
            Xt = np.vstack((Xt,yp))

    y = y.cpu().detach().numpy()
    yv = yv.cpu().detach().numpy()
    yt = yt.cpu().detach().numpy()

    #from sklearn.preprocessing import MinMaxScaler
    X_final = np.vstack((X,Xv))
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler.fit(X_final)
    #X_train_norm =  scaler.transform(X_final)

    from sklearn.datasets import dump_svmlight_file
    dump_svmlight_file(X_final,np.hstack((y,yv)),'traindata_'+str(item)+'.scl',zero_based=False)

    Xt_final = Xt
    #X_test_norm =  scaler.transform(Xt_final)
    dump_svmlight_file(Xt_final,yt,'testdata_'+str(item)+'.scl',zero_based=False)

