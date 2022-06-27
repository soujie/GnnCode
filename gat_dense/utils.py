import numpy as np 
import scipy.sparse as sp
import torch
import dgl 


def load_data():
    g = dgl.data.CoraGraphDataset()[0]
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    val_mask = g.ndata['val_mask']
    g = dgl.to_bidirected(g,copy_ndata=True) 
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    adj = g.adjacency_matrix().to_dense()
    labels = g.ndata['label']
    feats = g.ndata['feat']
    
    return adj , feats , labels , train_mask ,val_mask , test_mask
    

def accuracy(output:torch.Tensor,labels:torch.Tensor,mask:torch.Tensor):
    output = output[mask]
    labels = labels[mask]
    return (output == labels).sum()/len(output)
    
dgl.nn