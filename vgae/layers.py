import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    '''
    simple GCN layer
    '''
    def __init__(self, in_featrues, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_featrues = in_featrues
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_featrues, out_features))
        self.reset_parameter()

    def reset_parameter(self):
        torch.nn.init.kaiming_normal_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               

class InnerProductDecoder(Module):
    ''' decoder for using inner product for predict.'''
    
    def __init__(self,dropout,act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout=dropout
        self.act=act
        
    def forward(self,z):
        z=F.dropout(z,self.dropout,training=self.training)
        adj= self.act(torch.mm(z, z.t()))
        return adj