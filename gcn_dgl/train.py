import argparse
import dgl
import torch
import torch.nn.functional as F
from model import Gcn

def eval(model,graph,feats,label,mask):
    model.eval()
    pred= model(graph,feats)
    
    pred = pred[mask]
    label = label[mask]
    
    _,indices = torch.max(pred,dim=1)
    correct = torch.sum(indices==label)
    return correct.item()*1.0 / len(label)
    

def main(args):
    assert args.dataset =='cora' 
    data = dgl.data.CoraGraphDataset()
    g=data[0]
    
    features = g.ndata['feat']
    label = g.ndata['label']
    train_mask =g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    
    degree = g.in_degrees().float()
    norm = torch.pow(degree,-0.5)
    norm[torch.isinf(norm)]=0 
    
    g.ndata['norm']=norm.unsqueeze(1)
    
    model = Gcn(in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                activation=F.relu,
                dropout=args.dropout,
                layer_type=args.layer_type
                )
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    
    for epoch in range(args.epoches):
        model.train()
        
        pred = model(g,features)
        loss = loss_fn(pred[train_mask],label[train_mask])
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        train_acc=eval(model,g,features,label,train_mask)
        val_acc=eval(model,g,features,label,val_mask)
        
        print(f'Epoch {epoch} | Loss{loss.item()} | train acc{train_acc} | val acc{val_acc}')
        
    test_acc=eval(model,g,features,label,test_mask)
    print(f'Test Acc : {test_acc}')
    
    

if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--dropout',type=float , default=0.6,help='dropout prob')
    parse.add_argument('--dataset',type=str,default='cora',help='cora dateset')
    parse.add_argument('--lr',type=float,default=1e-2, help= ' learning rate')
    parse.add_argument('--epoches',type=int ,default=100, help='total train epoches')
    parse.add_argument('--n-hidden',type=int , default=16, help='hidden layer dim')
    parse.add_argument('--n-layers',type=int , default=1 , help='number of gcn layer , large layer number may lead over-smooth')
    parse.add_argument('--weight-decay',type=float,default=5e-4 , help='weight for l2 norm ')
    parse.add_argument('--self-loop',type=bool,default=False,help='add self loop ')
    parse.add_argument('--layer-type',type=str,default='raw',help='use  custome dgl / raw gcn bolck')
    args = parse.parse_args()
    print(args)
    
    main(args)