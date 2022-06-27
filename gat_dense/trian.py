from random import random
from onnxoptimizer import optimize
import torch
import argparse
import numpy as np
import random

from utils import load_data
from model import GAT

def eval(model,feats,adj,label,mask):
    model.eval()
    pred= model(feats,adj)
    
    pred = pred[mask]
    label = label[mask]
    
    _,indices = torch.max(pred,dim=1)
    correct = torch.sum(indices==label)
    return correct.item()*1.0 / len(label)

def train(args):
    adj,feats,labels,train_mask,val_mask,test_mask = load_data()
    
    model = GAT(n_feats=feats.shape[1],
                n_hidden=args.hidden,
                n_classes=int(labels.max().item())+1,
                dropout=args.dropout,
                n_heads=args.n_heads,
                alpha=args.alpha
                )
    optimize = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        model.train()
        pred = model(feats,adj)
        optimize.zero_grad()
        loss = loss_fn(pred[train_mask],labels[train_mask])
        loss.backward()
        optimize.step()
        
        train_acc = eval(model , feats ,adj ,labels, train_mask)
        val_acc = eval(model , feats ,adj , labels, val_mask)

        print(f'Epoch {epoch} | Loss{loss.item()} | train acc{train_acc} | val acc{val_acc}')
    
    print(f'Test acc: {eval(model,feats , adj,labels , test_mask)}')
    
    
    
        
        


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int, default= 20)
    parser.add_argument('--lr',type=float , default=5e-3)
    parser.add_argument('--weight_decay',type=float , default=5e-4)
    parser.add_argument('--hidden',type=int , default=8)
    parser.add_argument('--n-heads',type=int,default=8)
    parser.add_argument('--dropout',type=float,default=0.6)
    parser.add_argument('--alpha',type=float,default=0.2)
    
    args = parser.parse_args()
    seeds= 123
    
    np.random.seed(seeds)
    random.seed(seeds)
    torch.manual_seed(seeds)
    
    train(args)
    
    
    
    
    