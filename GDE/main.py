from operator import mod
import torch
import argparse
import os
import gc
import pandas as pd
import numpy as np
from layer import GDE_layer
from utils import checkAndGetNodeSize, convert2tensor, cal_idcg, cal_score
from tqdm import tqdm
from torch.utils.data import DataLoader


def main(args):
    train_df = pd.read_csv(f'../datasets/{args.dataset}/train_sparse.csv')
    train_len = train_df.shape[0]
    train_rating = torch.load(
        f'./cache/{args.dataset}/train_rate_tensor.pkl').to(args.device)
    test_adjlist = convert2tensor(args.dataset, 'test', 'adjlist')
    del train_df
    gc.collect()

    user_size, item_size = checkAndGetNodeSize(args.dataset)
    model = GDE_layer(user_size=user_size, item_size=item_size,h_dim=args.h_dim, beta=args.beta, feature_type=args.method,
                      dropout=args.dropout, reg=args.reg, task=args.dataset, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.reg,eps=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.reg,momentum=0.8)
    # train_loader = DataLoader(torch.LongTensor(range(user_size)),
    #                           args.batch_size, True)

    iters = train_len // args.batch_size
    
    
    best_score = np.inf
    patience = 0
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(range(iters))
        pbar.set_description(f'epoch:{epoch} :')
        for _ in pbar:
            u = torch.LongTensor(np.random.randint(0,user_size,args.batch_size))
            pos_samples = torch.multinomial(train_rating[u], 1).flatten()
            neg_samples = torch.multinomial(1 - train_rating[u], 1).flatten()
            loss = model(u, pos_samples, neg_samples,args.loss_type)

            assert ~torch.isnan(loss)  , print(loss)
            
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(),max_norm=20,norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.detach()
        print(f'epoch: {epoch} , train_loss: {total_loss/iters} ')
        ndcg10, ndcg20, recall10, recall20=val(model, test_adjlist, user_size,train_rating)
        # if (epoch) % 10 ==0:
        print(f'ndcg10:{ndcg10},ndcg20:{ndcg20},recall10:{recall10},recall20:{recall20}')
        if (curr_score:=(ndcg20+recall20))<best_score:
            best_score = curr_score
            patience = 0
            print(f'save best model at epoch {epoch}')
            torch.save(model,'./outputs/best.pt')
        else:
            patience+=1
            torch.save(model,'./outputs/curr.pt')
            if patience>args.patience:
                break



def val(model: GDE_layer, adjlist, user_size,train_mask):
    predict = model.predict_matrix()
    predict = predict * (1-train_mask)
    idcg = cal_idcg()
    ndcg10,ndcg20,recall10,recall20=0.0,0.0,0.0,0.0
    for i in range(user_size):
        test_size = len(adjlist[i])

        all10 = 10 if test_size > 10 else test_size
        all20 = 20 if test_size > 20 else test_size

        topn = predict[i].topk(20)[1]  #topk indices

        dcg10, dcg20, hit10, hit20 = cal_score(topn, adjlist[i])
        ndcg10 += (dcg10 / idcg[all10])
        ndcg20 += (dcg20 / idcg[all20])
        recall10 += (hit10 / all10)
        recall20 += (hit20 / all20)
    ndcg10, ndcg20, recall10, recall20 = round(ndcg10 / user_size, 4), round(
        ndcg20 / user_size, 4), round(recall10 / user_size,
                                      4), round(recall20 / user_size, 4)
    return ndcg10, ndcg20, recall10, recall20

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pinterest')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch-size', type=int, default=256,help='number of users will be used per epoch')
    parser.add_argument('--h-dim',
                        type=int,
                        default=64,
                        help='hidden dim of GNN model lookup embedding layer')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--reg',
                        type=float,
                        default=0.01,
                        help='l2-norm penalty parameter')
    parser.add_argument('--beta',
                        type=int,
                        default=3,
                        help='eigenvalue filter parameter')
    parser.add_argument(
        '--method',
        type=str,
        default='smooth',
        help='use  both smooth and rough eigenvalue or only use former ')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--loss-type',type=str,default='bpr')
    parser.add_argument('--patience',type=int,default=20)

    args = parser.parse_args()
    args.cache_file = os.path.abspath('./cache')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args)
    main(args)



