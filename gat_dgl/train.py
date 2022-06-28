import argparse
import dgl
import torch
from model import GAT
from dgl.data import CoraGraphDataset

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate(model,g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g,features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def main(args):
    g = CoraGraphDataset()[0]
    feat = g.ndata['feat']
    label = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = feat.shape[1]
    n_classes = max(label)+1
    
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    
    heads = ([args.num_heads]*(args.num_layers-1)) + [args.num_out_heads]
    model = GAT(args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                torch.nn.functional.elu,
                args.feat_dropout,
                args.attn_dropout,
                args.negative_slope,
                args.residual)

    print(model)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    
    for epoch in range(args.epochs):
        model.train()
        
        pred = model(g,feat)
        loss = loss_fn(pred[train_mask],label[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_acc = accuracy(pred[train_mask],label[train_mask])
        
        val_acc = evaluate(model,g,feat,label,val_mask)
        print(f'epoch {epoch} , loss {loss.item()}, train acc {train_acc} , val acc{val_acc}')
    
    test_acc = evaluate(model,g,feat,label,test_mask)
    print(f'test acc {test_acc}')
        
    

if __name__=='__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--num-heads',type=int,default=8)
    parser.add_argument('--num-out-heads',type=int,default=1,help='number of output attention heads, default 1')
    parser.add_argument('--num-layers',type=int,default=2)
    parser.add_argument('--num-hidden',type=int, default=8)
    parser.add_argument('--residual',action='store_true',default=False)
    parser.add_argument('--feat-dropout',type=float,default=.6)
    parser.add_argument('--attn_dropout',type=float,default=.6)
    parser.add_argument('--lr',type=float,default=5e-3)
    parser.add_argument('--weight-decay',type=float,default=5e-4)
    parser.add_argument('--negative-slope',type=float,default=0.2)
    args = parser.parse_args()
    
    main(args)
    
    
    