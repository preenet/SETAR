from pathlib import Path

import joblib
import numpy as np
import src.utilities as utils
import torch
from sklearn.model_selection import train_test_split
from transformers import CamembertTokenizer, RobertaModel

configs = utils.read_config()
root = utils.get_project_root()

Xo, yo = joblib.load(Path.joinpath(root, configs['data']['processed_kt_sav']))

class Camembert(torch.nn.Module):
    def __init__(self):
        super(Camembert, self).__init__()
        self.l1 = RobertaModel.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
        #for param in self.l1.parameters():
        #    param.requires_grad = False
        self.h1 = torch.nn.Linear(768, 50)
        self.a1 = torch.nn.Sigmoid()
        self.h2 = torch.nn.Linear(50, 4)
        #self.hidden_size = self.l1.config.hidden_size
        #self.LSTM = torch.nn.LSTM(input_size=self.hidden_size, num_layers=2, hidden_size=100, dropout=0.1, bidirectional=True)
        #self.hidden1 = torch.nn.Linear(200 , 100)
        #self.act1 = torch.nn.ReLU()
        #self.dropout = torch.nn.Dropout(0.2)
        #self.hidden2 = torch.nn.Linear(100, 4)
        #self.act2 = torch.nn.Softmax(-1)  // CrossEntropy already did it

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        X = output_1[0]
        X = self.h1(X[:,0])
        X = self.a1(X)
        X = self.h2(X)
        #X, (last_hidden, last_cell) = self.LSTM(X)
        #X, _ = torch.max(X, 1)
        #X = self.hidden1(X)
        #X = self.act1(X)
        #X = self.dropout(X)
        #X = self.hidden2(X)
        return X

    def extract(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        X = output_1[0]
        X = self.h1(X[:,0])
        X = self.a1(X)
        #X = self.h2(X)
        #X, (last_hidden, last_cell) = self.LSTM(X)
        #X, _ = torch.max(X, 1)
        #X = self.hidden1(X)
        #X = self.act1(X)
        #X = self.dropout(X)
        #X = self.hidden2(X)
        return X

SEED = [3]
EP = [8]
for iii, item in enumerate(SEED):
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
    
    model = Camembert()
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
    n_iters = EP[iii]
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

        print("epoch: "+str(epoch)+ " loss: "+str(train_loss/n_train) + "tr acc: "+str(train_correct/n_train) + " val acc: " +str(valid_correct/n_val))

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
    
    torch.save(model.state_dict(), "./model_"+str(item)+".pt")
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
    
    del model
    torch.cuda.empty_cache()
