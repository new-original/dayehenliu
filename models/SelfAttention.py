# -*- coding: utf-8 -*-
"""
Created on Fri May 10 10:05:23 2019

@author: 86156
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
cuda_device_id = '0'

class StructuredSelfAttention(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """
   
    def __init__(self,batch_size=16,input_dim=7*24,embedding_dim=64,lstm_hid_dim=128, \
                 d_a=100,r=10,max_len=26,num_class=9,use_clssify=True,dropout=0.5,lstm_layers=2):
        """
        Initializes parameters suggested in paper
 
        Args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            max_len     : {int} number of lstm timesteps
            emb_dim     : {int} embeddings dimension
            vocab_size  : {int} size of the vocabulary
            use_pretrained_embeddings: {bool} use or train your own embeddings
            embeddings  : {torch.FloatTensor} loaded pretrained embeddings
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes
 
        Returns:
            self
 
        Raises:
            Exception
        """
        super(StructuredSelfAttention,self).__init__()
        # input:[26,7*24]
        self.embedding = torch.nn.Linear(input_dim,embedding_dim)
        self.lstm_layers = lstm_layers
        # [b,max_len,embedding_dim]
        self.lstm = torch.nn.LSTM(embedding_dim,lstm_hid_dim,lstm_layers,batch_first=True,dropout=dropout)

        self.linear_first = torch.nn.Linear(lstm_hid_dim,d_a)
        self.linear_first.bias.data.fill_(0)
        # r:  want r different parts to be extracted from the sequence
        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)
        
        self.classification = torch.nn.Linear(lstm_hid_dim,num_class)

        self.batch_size = batch_size       
        self.max_len = max_len
        self.lstm_hid_dim = lstm_hid_dim
        self.hidden_state = self.init_hidden()
        self.r = r
        self.use_clssify = use_clssify
        self.dropout = torch.nn.Dropout(dropout)
       
        
    def init_hidden(self):
        device = torch.device("cuda:"+cuda_device_id if torch.cuda.is_available() else "cpu")
        w1 = Variable(torch.zeros(self.lstm_layers,self.batch_size,self.lstm_hid_dim)).to(device)
        w2 = Variable(torch.zeros(self.lstm_layers,self.batch_size,self.lstm_hid_dim)).to(device)
        return (w1,w2)
       
        
    def forward(self,x):  
        # x: (16,26,7,24)
        embeddings = self.dropout(self.embedding(x.view(self.batch_size,self.max_len,-1)))
        # embedings: (16,26,50)    (b,n,d)
        outputs, self.hidden_state = self.lstm(embeddings,self.hidden_state)   
        # outputs: (16,26,100),  (b,n,2u)  self.hidden_state: (1,16,100)   
        x = torch.tanh(self.dropout(self.linear_first(outputs)))
        # x: (16,26,200)  (b,n,da)
        x = self.linear_second(x)       
        # x: (16,26,15)   (b,n,r)
        x = F.softmax(x,1)      
        # x: (16,26,15)   (b,n,r)
        attention = x.transpose(1,2)    
        # attention: (16,15,26)  (b,r,n)     (b,r,n) * (b,n,2u)
        sentence_embeddings = attention@outputs   
        # sentence_embeddings: (16,26,100)  (b,r,2u)  r次关注的加权
        avg_embeddings = torch.sum(sentence_embeddings,1)/self.r
        # avg_embeddings: (16,100)     (b,2u)
        if self.use_clssify:
            out = self.classification(avg_embeddings)
            return out,attention
        else:
            return avg_embeddings,attention
       
	   
	#Regularization
    def l2_matrix_norm(self,m):
        """
        Frobenius norm calculation
 
        Args:
           m: {Variable} ||AAT - I||
 
        Returns:
            regularized value
        """
        return torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5)
    
if __name__ == '__main__':
    print('hello_model!')
    att_model = StructuredSelfAttention()
    input_tenor = torch.randn(16,26,7,24)
    out = att_model(input_tenor)
    print(out[0].shape)
    