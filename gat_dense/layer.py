import scipy.sparse as sp 
import torch
import torch.nn  as nn 
import torch.nn.functional as F 



class GatLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 dropout,
                 alpha,
                 concat = True
                 ) -> None:
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        

        self.W = nn.parameter.Parameter(torch.empty((in_feats,out_feats)))
        self.a = nn.parameter.Parameter(torch.empty((2*out_feats,1)))
        self.act = nn.LeakyReLU(alpha)
        self.reset_parameter()
        
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.W.data,gain=1.414)
        nn.init.xavier_uniform_(self.a.data,gain=1.414)
    
    def forward(self,adj,x):
        '''
        adj : (n x n)
        x   : (n x in_feats)
        '''
        h=torch.mm(x,self.W) # W : in_feats x out_feats ; h : n x out_feats
        e = self._call_attention_score(h)
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj>0 ,e , zero_vec)
        attention = F.softmax(attention,1) # n x n , 按照行进行softmax
        attention = self.dropout(attention) 
        
        h_prime = torch.matmul(attention,h) # n x out_feats
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
        
    def _call_attention_score(self,wh):
        '''
        原始的attention 机制为 a^T * (Wh1||Wh2)
        但上式等价于 a_1^T * Wh1 + a_2^T * Wh2 , 其中 a_1 为a 的前一半, a_2 为a 的后半部分
        
        Q,K,V = wh .
        '''
        
        wh1 = torch.matmul(wh,self.a[:self.out_feats])
        wh2 = torch.matmul(wh,self.a[self.out_feats:])
        # broad cast add 
        e = wh1 + wh2 
        return self.act(e)

    
    
