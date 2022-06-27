from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class GraphSage(nn.Module):
    def __init__(self,input_dim:int,hidden_dim:list,num_neighbors_list:list):
        """实现GraphSage层

        Args:
            input_dim (int): 输入维度
            hidden_dim (list): 隐藏层节点数
            num_neighbors_list (list): 各阶邻居采样数
        """        
        super().__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.num_neighbors_list=num_neighbors_list
        self.num_layers=len(num_neighbors_list)
        self.gcn=nn.ModuleList()
        self.gcn.append(SageGcn(input_dim,hidden_dim[0]))
        for idx in range(0,len(hidden_dim)-2):
            self.gcn.append(SageGcn(hidden_dim[idx],hidden_dim[idx+1]))
        self.gcn.append(SageGcn(hidden_dim[-2],hidden_dim[-1],act=None))
        
    def forward(self,node_features_list):
        """
        Args:
            node_features_list (List[List[tensor]]): 模型层数x采样数x特征数
        """        
        hidden=node_features_list #init feature list
        for l in range(self.num_layers): #forward
            next_hidden=[] 
            gcn=self.gcn[l]
            for hop in range(self.num_layers-l):  #依次聚合source节点及其各阶子节点的特征
                src_node_features=hidden[hop]
                src_node_num=len(src_node_features)
                neighbor_node_features=hidden[hop+1].view((src_node_num,self.num_neighbors_list[hop],-1))
                h=gcn(src_node_features,neighbor_node_features)
                next_hidden.append(h)
            hidden=next_hidden
        return hidden[0] #返回src 节点聚合后的节点特征
                


class SageGcn(nn.Module):
    def __init__(self,input_dim,hidden_dim,act=F.leaky_relu,
                 aggr_neighbor_method='mean',
                 aggr_hidden_method='sum',
                 ):
        """执行聚合操作

        Args:
            input_dim (int): 输入维度
            hidden_dim (int): 隐藏维度
            act (torch.nn.functional, optional): 激活函数. Defaults to F.leaky_relu.
            aggr_neighbor_method (str, optional): neighbor feature 聚合方式. Defaults to 'mean'.
            aggr_hidden_method (str, optional): 隐藏层特征聚合方式. Defaults to 'sum'.
        """        
        super().__init__()
        assert aggr_neighbor_method in ['mean','sum','max'] 
        assert aggr_hidden_method in ['sum','concat']
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.aggr_neighbor_method=aggr_neighbor_method
        self.aggr_hidden_method=aggr_hidden_method
        self.act=act
        self.aggregator=NeighborAggregator(input_dim, hidden_dim,
                                             aggr_method=aggr_neighbor_method)
        self.weight=nn.Parameter(torch.Tensor(input_dim,hidden_dim))
        self.reset_parameter()
    
    def reset_parameter(self):
        init.kaiming_normal_(self.weight)
    
    def forward(self,src_node_features,neighbor_node_features):
        neighbor_hidden=self.aggregator(neighbor_node_features)
        self_hidden=torch.matmul(src_node_features,self.weight)
        
        if self.aggr_hidden_method=='sum':
            hidden=self_hidden+neighbor_hidden
        elif self.aggr_hidden_method=='concat':
            hidden=torch.cat([self_hidden,neighbor_hidden],dim=1)
        
        if self.act:
            return self.act(hidden)
        else:
            return hidden
        
        
        
        
        

class NeighborAggregator(nn.Module):
    def __init__(self,input_dim,output_dim,use_bias=False,aggr_method='mean'):
        """执行邻居聚合操作

        Args:
            input_dim (int): 输入维度
            output_dim (int): 输出维度
            use_bias (bool, optional): 是否使用bias. Defaults to False.
            aggr_method (str, optional): 聚合方法, 目前只支持 ['mean','concat']. Defaults to 'mean'.
        """        
        super().__init__()
        assert aggr_method in ['mean','sum','max']
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.use_bias=use_bias
        self.aggr_aggr_method=aggr_method
        self.weight=nn.Parameter(torch.tensor(input_dim,output_dim)) 
        if self.use_bias:
            self.bias=nn.Parameter(torch.tensor(self.output_dim))
        self.reset_parameter()
    
    def reset_parameter(self):
        init.kaiming_normal_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)
    
    def forward(self,neighbor_feature):
        if self.aggr_aggr_method=='mean':
            aggr_neighbor=neighbor_feature.mean(dim=1)
        elif self.aggr_aggr_method=='sum':
            aggr_neighbor=neighbor_feature.sum(dim=1)
        elif self.aggr_aggr_method=='max':
            aggr_neighbor=neighbor_feature.max(dim=1)
        
        neighbor_hidden=torch.matmul(aggr_neighbor,self.weight)
        if self.use_bias:
            neighbor_hidden+=self.bias
        
        return neighbor_hidden
        