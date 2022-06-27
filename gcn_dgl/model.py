import torch
import torch.nn as nn 
from dgl.nn.pytorch import GraphConv
import dgl
from  typing import Optional
from layer import GCNLayer

class Gcn(nn.Module):
    layer_dict = {
        'raw' : GraphConv,
        'dgl' : GCNLayer
    }
    def __init__(self,
                 in_feats:int,
                 n_hidden:int,
                 n_classes:int,
                 n_layers:int,
                 activation:nn.Module,
                 layer_type:str,
                 dropout:Optional[float],
                ):
        '''
        基于dgl 提供的gcn layer 实现的gcn 模型
        '''
        super().__init__()
        self.layers = nn.ModuleList()
        
        assert layer_type in ['raw','dgl']
        gcn_block = self.layer_dict[layer_type]
        
        #add first gcn block 
        self.layers.append(gcn_block(in_feats,n_hidden,activation=activation))
        #add hidden gcn block
        for _ in range(n_layers-1):
            self.layers.append(gcn_block(n_hidden,n_hidden,activation=activation))
        #add output gcn block
        self.layers.append(gcn_block(n_hidden,n_classes))
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self,g,features):
        h=features
        for i,layer in enumerate(self.layers):
            if i!=0:
                h = self.dropout(h)
            h=layer(g,h)
        return h
        


