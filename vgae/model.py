import torch 
import torch.nn as nn
import torch.nn.functional as F 

from layers import GraphConvolution,InnerProductDecoder

class Vgae(nn.Module):
    def __init__(self,input_dim,hidden_dim1,hidden_dim2,dropout):
        super(Vgae, self).__init__()
        self.gc1=GraphConvolution(input_dim,hidden_dim1,dropout,act=F.relu)
        self.gc2=GraphConvolution(hidden_dim1,hidden_dim2,dropout,act=lambda x:x)
        self.gc3=GraphConvolution(hidden_dim1,hidden_dim2,dropout,act=lambda x:x)
        self.dc=InnerProductDecoder(dropout,act=lambda x:x)

    def encode(self,x,adj):
        hidden1=self.gc1(x,adj)
        return self.gc2(hidden1,adj),self.gc3(hidden1,adj)
    
    def reparameterize(self,mu,logvar):
        if self.training:
            std=torch.exp(logvar)
            eps=torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self,x,adj):
        mu,logvar=self.encode(x, adj) #实现权重共享
        z=self.reparameterize(mu, logvar)
        return self.dc(z),mu,logvar
        