#!/usr/bin/env python

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''

@File    : graphSAGE_subgraph.py
@Time    : 2022/01/04 16:31:39
@Author  : Ian Yang
@Contact : 997417824@qq.com
@Version : 0.1
@License : Apache License Version 2.0, January 2004
@Desc    : None

TODO:
    1. 将伪代码改写
'''

import dgl #开源gnn 模型
import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.data import Dataset

def build_graph_test():
    '''
    生成测试小图
    ---------------
    Return:
        dgl.DGLGraph
    '''
    src_nodes=torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6])
    dst_nodes=torch.tensor([1, 2, 0, 2, 0, 1, 3, 4, 5, 6, 2, 3, 3, 3])
    graph=dgl.graph((src_nodes,dst_nodes))
    return graph


def build_cora_dataset(add_symmertric_edges=True,add_self_loop=True):
    dataset=dgl.data.CoraGraphDataset()
    graph=dataset[0]

    if add_symmertric_edges:
        edges=graph.edges()
        graph.add_edges(edges[1],edges[0])
        
    graph=dgl.remove_self_loop()
    if add_self_loop:
        graph=dgl.add_self_loop()
    return graph

class HomoNodesSet(Dataset):
    def __init__(self,g:dgl.DGLGraph,mask):
        self.g=g
        self.nodes=g.nodes()[mask].tolist()
        
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, index):
        return self.nodes[index]

class NodeGraphCollactor(object):
    ''' 
    select heads/tails/neg_tails's neighbors for aggregation
    '''
    
    def __init__(self,g,neighbors_every_layer=[5,1]):
        self.g=g
        self.neighbors_every_layer=neighbors_every_layer
        
    def __call__(self,batch):
        blocks,seeds=self.sample_blocks(batch)
        return blocks,seeds
    def sample_blocks(self,seeds):
        blocks=[]
        for n_neighbors in self.neighbors_every_layer:
            frontier=dgl.sampling.sample_neighbors(slef.g,seeds,fanout=n_neighbors,edge_dir='in')
            block= slef.compact_and_copy(frontier,seeds)
            seeds=block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks,seeds
    
    def compact_and_copy(self,frontier,seeds):
        block=dgl.to_block(frontier,seeds)
        for col , data in frontier.edata.items():
            if col==dgl.EID:
                continue
            block.edata[col]=data[block.edata[EID]]
        return block
    
    
class WeightedSAGEConv(nn.Module):
    def __init__(self,input_dims,output_dims,act=f.relu,dropout=.5,bias=True):
        super().__init__()
        
        self.act=act
        self.Q=nn.Linear(input_dims, output_dims)
        self.W=nn.Linear(input_dims+output_dims, output_dims)
        
        if bias:
            self.bias=nn.Parameter(torch.FloatTensor(output_dims))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        'init Q,W and bias'
        pass
    
    def forward(self,g,h,weight=None):
        h_src,h_dst=h 
        with g.local_scope():
            g.srcdata['n'] = self.Q(h_src)
            g.update_all(fn.copy_src('n', 'm'), fn.mean('m', 'neigh'))  # aggregation or pool:fn.max()
            n = g.dstdata['neigh']
            z = self.act(self.W(torch.cat([n, h_dst], 1))) + self.bias
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z
        
class SAGENet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dims,
                 n_layers, act=F.relu, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(WeightedSAGEConv(input_dim, hidden_dims, act, dropout))
        for _ in range(n_layers - 2):
            self.convs.append(WeightedSAGEConv(hidden_dims, hidden_dims,
                                               act, dropout))
        self.convs.append(WeightedSAGEConv(hidden_dims, output_dims,
                                           act, dropout))
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.convs, blocks)):
            h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
            h = layer(block, (h, h_dst))
            if l != len(self.convs) - 1:
                h = self.dropout(h)
        return h
    
    
    
    dgl.