from turtle import forward
from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torch.nn.functional as F 
from layer import GatLayer


class GAT(nn.Module):
    def __init__(self,
                 n_feats,
                 n_hidden,
                 n_classes,
                 dropout,
                 alpha,
                 n_heads) -> None:
        super().__init__()
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        
        self.attentions = [GatLayer(n_feats,n_hidden,dropout,alpha,True) for _ in range(n_heads)]
        
        for i,layer in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i),layer)
        
        self.out_att = GatLayer(n_hidden*n_heads , n_classes, dropout,alpha,False)
        self.act = nn.ELU()
    
    def forward(self,x,adj):
        x = self.dropout1(x)
        x = torch.concat([att(adj,x) for att in self.attentions],dim=1) # 按照特征维度进行拼接
        x = self.dropout2(x)
        x = self.act(self.out_att(adj,x))
        return F.log_softmax(x,dim=1)
        