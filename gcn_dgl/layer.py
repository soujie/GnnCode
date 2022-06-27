from turtle import forward
from matplotlib.colors import NoNorm
import torch
import torch.nn as nn 
from torch.nn.parameter import Parameter
import dgl


'''
对于gcn layer, 其向量表达  \sigma(D^{-1/2}AD^{-1/2}*X*W)  , 
改写为, 对于第i个节点, 其信号
\sum_{j} w_{ij}A_{ij} h_i  = \sum_{j} 1/(\sqrt{d_i}) * 1/(\sqrt{d_j}) A_{ij} h_i, 
其中 h_i 为X * W 的第i行
按照消息传播框架, 其可分解为如下步骤:
1. 定义信息:
    h = X * W 
2. 进行信息传播, 具体来说就是计算 D^{-1/2}*h 部分:
    m = D^{-1/2}*h
3. 信息聚合, 通过mailbox 收到的邻居节点信息进行聚合, 计算 D^{-1/2}AD^{-1/2}*X*W  整体.

4. 更新节点信息:
    施加激活函数和增加bias 项
'''

def gcn_msg(edge):
    ''' 
    消息传播函数
    '''
    msg = edge.src['h']*edge.src['norm'] # 按照行进行广播
    return {'m':msg}

def gcn_reduce(node):
    '''
    聚合函数
    '''
    accum = torch.sum(node.mailbox['m'],1)*node.data['norm']
    return {'h':accum}
    
class NodeApplyModule(nn.Module):
    def __init__(self,out_feats,activation=None,bias=True):
        '''
        更新函数
        '''
        super().__init__()
        if bias :
            self.bias = nn.parameter.Parameter(torch.Tensor(out_feats))
        else :
            self.bias = None
        self.activation = activation
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.bias is not None:
            nn.init.uniform_(self.bias)
    def forward(self,nodes):
        h = nodes.data['h']
        if self.bias is not None:
            h = h+ self.bias
        if self.activation:
            h = self.activation(h)
        return {'h':h}

class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation=None,
                 dropout=False,
                 bias=True):
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_feats,out_feats))
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout=0
        
        self.node_update=NodeApplyModule(out_feats,activation,bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self,g:dgl.graph
                ,h:torch.Tensor):
        if self.dropout:
            h=self.dropout(h)
        g.ndata['h']=torch.mm(h,self.weight)
        g.update_all(gcn_msg,gcn_reduce,self.node_update)
        h=g.ndata.pop('h')
        return h