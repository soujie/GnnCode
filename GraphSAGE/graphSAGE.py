#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : graphSAGE.py
@Time    : 2021/12/23 09:57:28
@Author  : Ian Yang
@Contact : 997417824@qq.com
@Version : 0.1
@License : Apache License Version 2.0, January 2004
@Desc    : None
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn.init as init 



def sampling(src_nodes,sample_num:int,neighbor_table:dict):
    '''
    根据源节点采样指定数量的邻居节点,注意使用的是有放回采样.
    当某个节点的邻居节点数量少于采样数量时,采样结果出现重复的节点.
    ---------------
    Args:
         src_nodes {list,ndarray} 源节点列表
         sample_num  {int} 需要采样的节点数
         neighbor_table {dict} 节点到其邻居节点的映射表
    Return:
         ndarray 采样结果构成的列表
    '''
    results=[]
    for sid in src_nodes:
        #从节点的邻居中进行有放回的采样
        res=np.random.choice(neighbor_table[sid],size=(sample_num, ))
        results.append(res)
    return np.asarray(results).flatten()

def multihop_sampling(src_nodes,sample_nums,neighbor_table):
    '''
    根据源节点进行多阶采样,采样结果仅为节点ID
    ---------------
    Args:
         src_nodes  {list,np.ndarray}   源节点id
         sample_nums  {list of int }   每一阶需要采样的个数
         neighbor_table {dict}节点到其邻居的映射
    Return:
         [list of ndarray] 每一阶采样的结果
    '''
    sampling_result=[src_nodes]
    for k,hopk_num in enumerate(sample_nums):
        hopk_result=sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result

class NeighborAggregator(nn.Module):
    def __init__(self,input_dim,output_dim,use_bias=False,aggr_method='mean'):
        '''
        聚合邻居方式的实现
        ---------------
        Args:
             input_dim  {int}   输入特征的维度
             output_dim  {int}   输出特征的维度
             use_bias   {bool}  是否使用截距,默认为False
             aggr_method    {string} 聚合方式, 默认为mean
        Return:
             pass
        '''
        super(NeighborAggregator,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.use_bias=use_bias
        self.aggr_method=aggr_method
        self.weight=nn.Parameter(torch.Tensor(input_dim,output_dim))
        if self.use_bias:
            self.bias=nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameter()
        
    def reset_parameter(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)
            
    def forward(self,neighbor_feature):
        if self.aggr_method=='mean':
            aggr_neighbor=neighbor_feature.mean(dim=1)
        elif self.aggr_method=='sum':
            aggr_neighbor=neighbor_feature.sum(dim=1)
        elif self.aggr_method=='max':
            aggr_neighbor=neighbor_feature.max(dim=1)
        else:
            raise ValueError('Unknown aggr type , excepted sum ,max, or mean , but got {}'.format(self.aggr_method))
        neighbor_hidden=torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden+=self.bias
        return neighbor_hidden
    
class SageGCN(nn.Module):
    def __init__(self,input_dim,hidden_dim,activation=F.relu,aggr_neighbor_method='mean',aggr_hidden_method='sum'):
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ['mean','sum','max']
        assert aggr_hidden_method in ['sum','concat']    
        self.aggr_neighbor=aggr_neighbor_method
        self.aggr_hidden=aggr_hidden_method
        self.activation=activation
        self.aggregator=NeighborAggregator(input_dim, output_dim,aggr_method=aggr_neighbor_method)
        self.weight=nn.Parameter(torch.Tensor(input_dim,output_dim))
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        
    def forward(self,src_node_features, neighbor_node_features):
        neighbor_hidden=self.aggregator(neighbor_node_features)
        self_hidden=torch.matmul(src_node_features,self.weight)
        
        if self.aggr_hidden=='sum':
            hidden=self_hidden+neighbor_hidden
        elif self.aggr_hidden=='concat':
            hidden=torch.cat([self_hidden,neighbor_hidden],dim=1)
        else:
            raise ValueError('Except sum or concat , got {}'.format(self.aggr_hidden))
        
        if self.activation:
            return self.activation
        else:
            return hidden
        
        
class GraphSage(nn.Module):
    def __init__(self,input_dim,hidden_dim=[64,64],num_neighbors_list=[10,10]):
        super(GraphSage,self).__init__()
        self.input_dim=input_dim
        self.num_neighbors_list=num_neighbors_list
        self.num_layers=len(num_neighbors_list)
        self.gcn=[]
        self.gcn.append(SageGCN(input_dim,hidden_dim[0]))
        self.gcn.append(SageGCN(hidden_dim[0],hidden_dim[1],activation=None))
    def forward(self,node_features_list):
        hidden=node_features_list
        for l in range(self.num_layers):
            next_hidden=[]
            gcn=self.gcn[l]
            for hop in range(self.num_layers-1):
                src_node_features=hidden[hop]
                src_node_num= len(src_node_features)
                neighbor_node_features=hidden[hop+1].view(src_node_num, self.num_neighbors_list[hop],-1)
                h=gcn(src_node_features,neighbor_node_features)
                next_hidden.append(h)
            hidden=next_hidden
        return hidden[0]
    
            
            
        