import torch
from transformers import CamembertTokenizer, RobertaModel


class Camembert(torch.nn.Module):
    
    def __init__(self):
        super(Camembert, self).__init__()
        self.l1 = RobertaModel.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
        #for param in self.l1.parameters():
        #    param.requires_grad = False
        self.h1 = torch.nn.Linear(768, 50)
        self.a1 = torch.nn.Sigmoid()
        self.h2 = torch.nn.Linear(50, 3)
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
