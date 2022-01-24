import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize
from collections import namedtuple
import torch.nn as nn


def correct(output,target,std_ret,mean_ret):
    o=output.flatten()*std_ret +mean_ret
    t=target*std_ret + mean_ret
    tmp=o*t
    correct=0
    for i in range(len(tmp)):
        if(tmp[i].item() > 0):
            correct+=1
    return correct

def hit_profit(output,target,std_ret,mean_ret):
    o=output.flatten()*std_ret +mean_ret
    t=target*std_ret + mean_ret
    tmp=o*t
    hit_profit=0
    for i in range(len(tmp)):
        if(tmp[i].item() > 0):
            hit_profit+=torch.abs(target[i])
    return hit_profit

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long).view(-1)


def return_fun(data,delta_t,index):
    data1=data.to_numpy()
    ret_fun=[]
    for i in range(len(data1)-delta_t):
        ret_fun.append(100*(data1[i+delta_t,index]-data1[i,index])/data1[i,index])
    for i in range(delta_t):
        ret_fun.append(None)
    ret_fun=ret_fun
    return ret_fun


class LSTMNumeric(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.lstm=torch.nn.LSTM(input_size,hidden_size,num_layers=3,batch_first=True)
        self.fully_connected=torch.nn.Sequential(
            torch.nn.Linear(hidden_size*5,150),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(150,50),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(50,1)
        )
    def forward(self,x):
        batch_size=x.shape[0]
        hidden_size=30
        h0=torch.zeros(3,batch_size,hidden_size).cuda()
        c0=torch.zeros(3,batch_size,hidden_size).cuda()

        lstm_output, cels = self.lstm(x,(h0,c0))
        flatten_output=torch.flatten(lstm_output,start_dim=1)
        
        output=self.fully_connected(flatten_output)
        
        return output
    
    
class TextOnlyGRU(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,output_size):
        super(TextOnlyGRU,self).__init__()
        
        self.embedding=nn.Embedding(vocab_size,embedding_size)
        self.gru=nn.GRU(embedding_size, hidden_size,2, batch_first=True)
        self.fully_connected=nn.Sequential(
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,32),
            nn.Tanh(),
            nn.Linear(32,1)
        )
    def forward(self,x):
        embed=self.embedding(x)
        gru_output, h=self.gru(embed)
        g=gru_output[:,-1,:]
        return self.fully_connected(g)
    
    
class TextAndNumeric(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,numeric_size):
        super(TextAndNumeric,self).__init__()
        
        self.embedding=nn.Embedding(vocab_size,embedding_size)
        self.rnn_text=nn.GRU(embedding_size,hidden_size, 2 , batch_first=True)
        self.rnn_numeric=nn.LSTM(numeric_size, hidden_size, 2 , batch_first=True, bidirectional=True)
        self.numeric_fc=nn.Linear(10*hidden_size,hidden_size)
        self.fully_connected=nn.Sequential(
            nn.Linear(2*hidden_size,64),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(64,32),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(32,1)
        )
        
    def forward(self,headline,numeric):
        #text
        embed=self.embedding(headline)
        gru_output, h=self.rnn_text(embed)
        text_output=gru_output[:,-1,:]
        
        #numeric
        lstm_output, h = self.rnn_numeric(numeric)
        flatten_output=torch.flatten(lstm_output,start_dim=1)
        numeric_output=self.numeric_fc(flatten_output)
        
        #concatenate text and numeric
        hidden=torch.cat([text_output,numeric_output],1)
        output=self.fully_connected(hidden)
        
        return output
    
    
def test_tan(model,loader):
    loss_fn=torch.nn.MSELoss()

    tan.eval()
    with torch.no_grad():
        val_loss_buffer=[]
        val_ac=0
        cor=0
        hit_profit=0
        for i, (x,y) in enumerate(loader):
            headline,numeric = x
            numeric=numeric.float()
            headline=headline.cuda()
            numeric=numeric.cuda()
            y=y.cuda()
            
            output=model(headline,numeric)
            output.float()
            y.float()
            val_loss_buffer.append(loss_fn(output,y).item())
            if(correct(output, y, std_ret, mean_ret)==1):
                hit_profit += correct(output, y, std_ret, mean_ret) * abs(y*std_ret - mean_ret)
            else:
                hit_profit-= abs(y*std_ret - mean_ret)
            
            cor+=correct(output,y,std_ret,mean_ret)
            
        val_acc=cor/len(loader)
        val_loss=torch.mean(torch.tensor(val_loss_buffer))
        print(f"Test loss: {val_loss:.3f} Test Acc {val_acc:.5f}")
        print(f"Test Hit Profit: {hit_profit.item():0.3f}")
        
        return val_loss,val_acc,hit_profit