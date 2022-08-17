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

Xo, yo = joblib.load(Path.joinpath(root, configs['data']['kaggle_ws']))
# data = joblib.load(Path.joinpath(root, configs['data']['kaggle_to']))
# Xo = data[0]
# yo = data[1]

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

EP = 15
for item in range(0, 9):
    print("SEED:", item)
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
    
    model = Camembert(4)
    model.to("cuda")
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
    train_dl = DataLoader(TensorDataset(X1, X2, y), batch_size=32, shuffle=True)
    val_dl = DataLoader(TensorDataset(Xv1, Xv2, yv), batch_size=32, shuffle=True)
    
    # training
    lr = 1e-5
    n_iters = EP
    #eps = 1e-7
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(n_iters):
        model.train()
        train_loss = 0.0
        train_correct = 0.0
        valid_correct = 0.0
        n_train = 0.0
        for input1, input2, targets in train_dl:
            input1_gpu = input1.to('cuda', dtype = torch.long)
            input2_gpu = input2.to('cuda', dtype = torch.long)
            targets_gpu = targets.to('cuda', dtype = torch.long)
            optimizer.zero_grad()
            yp = model(input1_gpu, input2_gpu)
            l= loss(yp, targets_gpu)
            train_loss += l.item()
            train_correct += (torch.argmax(yp, 1) == targets_gpu).sum().item()
            n_train += len(targets)
            l.backward()
            optimizer.step()

        model.eval()
        n_val = 0.0
        with torch.no_grad():
            for val1, val2, tval in val_dl:
                n_val += len(tval)
                val1_gpu = val1.to('cuda', dtype = torch.long)
                val2_gpu = val2.to('cuda', dtype = torch.long)
                tval_gpu = tval.to('cuda', dtype = torch.long)
                yval = model(val1_gpu, val2_gpu)
                valid_correct += (torch.argmax(yval, 1) == tval_gpu).sum().item()

        print("epoch: "+str(epoch)+ " loss: "+str(train_loss/n_train) + " ,tr_acc: "+str(train_correct/n_train) + " ,val_acc: " +str(valid_correct/n_val))

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
            
    model_file = "./model_"+str(item)+".pt"
    torch.save(model.state_dict(), model_file)
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
    
    file = open(configs['output_scratch'] +"wangchan_10repeated_ws.csv", "a")
    
    torch.cuda.empty_cache()
    model = Camembert(4)
    model.load_state_dict(torch.load(model_file ))
    model.to('cuda')
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
    acc, pre, rec, mcc, auc, f1 = test_multi(yp, yv)            
    #acc, pre, rec, mcc, auc, f1 = test_binary(yp, yv)   
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
    acc, pre, rec, mcc, auc, f1 = test_multi(yp, yt)  
    #acc, pre, rec, mcc, auc, f1 = test_binary(yp, yt)  
    file.write("," + str(item) + "," +str(acc) + "," + str(pre) + "," + str(rec) + "," + str(mcc) + "," + str(auc) + "," + str(f1) +"\n") 
       
    del model
    torch.cuda.empty_cache()
