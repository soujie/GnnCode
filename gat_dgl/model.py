from typing import List
import torch
import torch.nn as nn 
import dgl.function as fn 
from layer import GatLayer

class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 hid_dim,
                 num_classes,
                 n_heads:List[int],
                 activation,
                 feat_dropout,
                 attn_dropout,
                 negative_slope,
                 residual
                 ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        
        if num_layers>1:
            self.gat_layers.append(GatLayer(in_dim,hid_dim,n_heads[0],feat_dropout,attn_dropout,negative_slope,False,self.activation))
            
            for l in range(1,num_layers-1):
                self.gat_layers.append(GatLayer(hid_dim*n_heads[l-1],hid_dim,n_heads[l],feat_dropout,attn_dropout,negative_slope,residual,self.activation))
            
            self.gat_layers.append(GatLayer(hid_dim*n_heads[-2],num_classes,n_heads[-1],feat_dropout,attn_dropout,negative_slope,residual,None))
        else:
            self.gat_layers.append(GatLayer(in_dim,num_classes,n_heads[0],feat_dropout,attn_dropout,negative_slope,residual,None))
        
    def forward(self,g,feat:torch.tensor):
        '''
        各gat 层的输出为 num_nodes x heads x out_dim 
        对于最后一层之前的, 将其按照head 通道flatten 为 num_nodes x (heads*out_dim) 进行表示
        而对于最后层, 将其按照head 通道取 mean 进行聚合
        '''
        h = feat
        for l  in range(self.num_layers):
            h = self.gat_layers[l](g,h)
            h = h.flatten(1) if l != self.num_layers-1 else h.mean(1)       
        return h 