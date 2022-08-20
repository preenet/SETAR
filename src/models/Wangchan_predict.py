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
from src.models.camerbert import Camembert
from transformers import CamembertTokenizer, RobertaModel

configs = utils.read_config()
root = utils.get_project_root()

######################################################################
model_path = str(Path.joinpath(root, configs['wangchan_models']['ws']))
num_class = 4
out_file_name = 'wangcha_10repeated_to_final.csv'
#Xo, yo = joblib.load(Path.joinpath(root, configs['data']['kaggle_ws']))

data = joblib.load(Path.joinpath(root, configs['data']['kaggle_ws']))
Xo = data[0]
yo = data[1]
######################################################################

def test_binary(yp, yt):     
    test_y = yt
    p = yp.argmax(1)
    pr = torch.nn.functional.softmax(torch.tensor(yp), dim=1)
    ACC = accuracy_score(test_y,p)
    SENS = precision_score(test_y,p)
    SPEC = recall_score(test_y,p)
    MCC = matthews_corrcoef(test_y,p)
    AUC = roc_auc_score(test_y,pr[:,1])
    F1 = 2*SENS*SPEC/(SENS+SPEC)
    return ACC, SENS, SPEC, MCC, AUC, F1

def test_multi(yp, yt):     
    test_y = yt
    p = yp.argmax(1)
    pr = torch.nn.functional.softmax(torch.tensor(yp), dim=1)
    ACC = accuracy_score(test_y,p)
    SENS = precision_score(test_y,p, average='macro')
    SPEC = recall_score(test_y,p, average='macro')
    MCC = matthews_corrcoef(test_y,p)
    AUC = roc_auc_score(test_y,pr,multi_class='ovo',average='macro')
    F1 = 2*SENS*SPEC/(SENS+SPEC)
    return ACC, SENS, SPEC, MCC, AUC, F1

SEED = [i for i in range(0,10)]

for idx, item in enumerate(SEED):
    print("SEED:", item)
    X_train, X_tmp, y, y_tmp = train_test_split(Xo, yo, test_size=0.4, random_state=item, stratify=yo)
    X_val, X_test, yv, yt = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=item, stratify=y_tmp)
  
    train_wisesight = X_train
    validation_wisesight = X_val
    test_wisesight = X_test
        
    wangchan_tokenizer = CamembertTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased", use_fast=True)
    
    # Model parameter
    MAX_LEN = 256

    model_file = model_path + "./model_"+str(item)+".pt"
    model = Camembert(num_class)
    model.load_state_dict(torch.load(model_file))
    model.to('cuda')
    
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
    train_dl = DataLoader(TensorDataset(X1, X2, y), batch_size=32, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xv1, Xv2, yv), batch_size=32, shuffle=True)


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
    
    file = open(configs['output'] + out_file_name, "a")
    
    torch.cuda.empty_cache()

    with torch.no_grad():
        for i, (input1, input2, targets) in enumerate(val_dl):
            input1_gpu = input1.to('cuda', dtype = torch.long)
            input2_gpu = input2.to('cuda', dtype = torch.long)
            targets_gpu = targets.to('cuda', dtype = torch.long)
            ytmp = model(input1_gpu, input2_gpu).cpu().detach().numpy() 
            if i == 0:
                yp = ytmp
            else:
                yp = np.vstack((yp, ytmp))
    if num_class > 2:
        acc, pre, rec, mcc, auc, f1 = test_multi(yp, yv) 
    else:          
        acc, pre, rec, mcc, auc, f1 = test_binary(yp, yv)   
    file.write(str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1))
    
    
    with torch.no_grad():
        for i, (input1, input2, targets) in enumerate(test_dl):
            input1_gpu = input1.to('cuda', dtype = torch.long)
            input2_gpu = input2.to('cuda', dtype = torch.long)
            targets_gpu = targets.to('cuda', dtype = torch.long)
            ytmp = model(input1_gpu, input2_gpu).cpu().detach().numpy() 
            if i == 0:
                yp = ytmp
            else:
                yp = np.vstack((yp, ytmp)) 
    if num_class > 2:
        acc, pre, rec, mcc, auc, f1 = test_multi(yp, yt) 
    else:          
        acc, pre, rec, mcc, auc, f1 = test_binary(yp, yt)    
    file.write("," + str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1) +"\n") 
       
    del model
    torch.cuda.empty_cache()
    file.close()
