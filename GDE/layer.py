import gc
from inspect import GEN_CLOSED
from operator import mod
import os 
import torch 
import torch.nn as nn

cache_file = os.path.abspath('./cache') 

class GDE_layer(nn.Module): 
    def __init__(self,
                 user_size,
                 item_size,
                 h_dim = 64,
                 beta=5,
                 feature_type = 'smooth',
                 dropout = 0.1,
                 reg =0.01, # l2-norm penalty 
                 task = 'ml_100k',
                 device = 'cpu'
                 ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(user_size,h_dim)
        self.item_embedding = nn.Embedding(item_size,h_dim)
        
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        
        self.beta = beta 
        self.reg =reg 
        if dropout !=0:
            self.dropout = nn.Dropout(dropout)
        assert feature_type in ['smooth','both'] , f'only support smooth / both method'
        if feature_type == 'smooth':
            # 只使用光滑超图时, 加载部分特征即可
            user_filter  = self.weight_feature(
                torch.Tensor(torch.load(os.path.join(cache_file,f'{task}/{task}_smooth_user_values.pt'))).to(device))
            item_filter = self.weight_feature(
                torch.Tensor(torch.load(os.path.join(cache_file,f'{task}/{task}_smooth_item_values.pt'))).to(device))
            user_vector = torch.Tensor(torch.load(os.path.join(cache_file,f'{task}/{task}_smooth_user_features.pt'))).to(device)
            item_vector = torch.Tensor(torch.load(os.path.join(cache_file,f'{task}/{task}_smooth_item_features.pt'))).to(device)
        else:
            # 同时使用smooth+ rough 时, 全量加载
            user_filter = torch.cat([
                self.weight_feature(torch.Tensor(torch.load(os.path.join(cache_file,f'{task}/{task}_smooth_user_values.pt'))).to(device)),
                self.weight_feature(torch.Tensor(torch.load(os.path.join(cache_file,f'{task}/{task}_rough_user_values.pt'))).to(device))
            ])
            item_filter = torch.cat([
                self.weight_feature(torch.Tensor(torch.load(os.path.join(cache_file,f'{task}/{task}_smooth_item_values.pt'))).to(device)),
                self.weight_feature(torch.Tensor(torch.load(os.path.join(cache_file,f'{task}/{task}_rough_item_values.pt'))).to(device))
            ])
            user_vector = torch.cat([
                torch.Tensor(torch.load(os.path.join(cache_file,f'{task}/{task}_smooth_user_features.pt'))).to(device),
                torch.Tensor(torch.load(os.path.join(cache_file,f'{task}/{task}_rough_user_features.pt'))).to(device)
            ],dim=1)
            item_vector = torch.cat([
                torch.Tensor(torch.load(os.path.join(cache_file,f'{task}/{task}_smooth_item_features.pt'))).to(device),
                torch.Tensor(torch.load(os.path.join(cache_file,f'{task}/{task}_rough_item_features.pt'))).to(device)
            ],dim=1)
        
        # 计算 (12) 式
        self.Lu = (user_vector*user_filter).mm(user_vector.t())
        self.Li = (item_vector*item_filter).mm(item_vector.t())
        
        del user_filter,item_filter,user_vector,item_vector
        gc.collect()
            
            
    def weight_feature(self,value):
        '''
        according equation (16) calculate gamma function
        TODO: add node degreee into gamma function for eval model                    
        '''
        return torch.exp(self.beta*value)
    
    def forward(self,
                user, # user list
                pos_item,
                neg_item,
                loss_type = 'adaptive'
                ):
        batch_size = user.shape[0]
        if hasattr(self,'dropout'):
            user_embed =self.dropout(self.Lu[user]).mm(self.user_embedding.weight)
            pos_item_embed = self.dropout(self.Li[pos_item]).mm(self.item_embedding.weight)
            neg_item_embed = self.dropout(self.Li[neg_item]).mm(self.item_embedding.weight)
        else:
            user_embed =self.Lu[user].mm(self.user_embedding.weight)
            pos_item_embed = self.Li[pos_item].mm(self.item_embedding.weight)
            neg_item_embed = self.Li[neg_item].mm(self.item_embedding.weight)
        
        res_neg = (user_embed*neg_item_embed).sum(1)
        res_pos = (user_embed*pos_item_embed).sum(1)
            
        if loss_type == 'adaptive':
            weight_neg = (1-(1-res_neg.sigmoid().clamp(max=0.99)).log10()).detach()
            out = (res_pos-weight_neg*res_neg).sigmoid()
        elif loss_type == 'bpr':
            # BPR loss
            out = (res_pos-res_neg).sigmoid()
        else:
            raise ValueError(f'{loss_type} only support adaptive or bpr')
        # regular_term = self.reg*(user_embed**2+pos_item_embed**2+neg_item_embed**2).sum()
        # return (-torch.log(out).sum()+regular_term)/batch_size
        return (-torch.log(out).sum())/batch_size
    
    def predict_matrix(self):
        user_embed = self.Lu.mm(self.user_embedding.weight)
        item_embed = self.Li.mm(self.item_embedding.weight)    
        return (user_embed.mm(item_embed.t())).sigmoid()

if __name__=='__main__':
    from utils import checkAndGetNodeSize
    user_size , item_size = checkAndGetNodeSize('ml_100k')
    model = GDE_layer(user_size,item_size)
    u = torch.LongTensor([1])
    pos = torch.LongTensor([2])
    neg = torch.LongTensor([3])
    print(model(u,pos,neg))