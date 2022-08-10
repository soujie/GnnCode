import torch
import gc 
import pandas as pd 
import argparse
import os
import scipy.sparse as sp 
from utils import checkAndGetNodeSize
import numpy as np 
 
user_size=None
item_size=None
dataset=None

smooth_ratio = 0.2 # only top 20% eigenvalues will be choosed to create smooth graph
rough_ratio = 0.002 # only last 0.2% eigenvalues will be choosed to create rough graph

cache_file = os.path.abspath('./cache')
def cal_spectral_feature(Adj:torch.tensor,#邻接矩阵
                         size:int,#topk 个
                         side:str = 'user', # 生成用户、产品侧
                         largest:bool = True, # 提取Smooth、rough 子图
                         niter = 5
                         ):
    '''
    通过Lobpcg 算法 获取前(后)k 个特征向量、特征值
    '''
    value,vector = torch.lobpcg(Adj,k=size,largest=largest,niter=niter)
    
    if largest:
        feature_file_name = os.path.join(cache_file,f'{dataset}_smooth_{side}_features.pt')
        value_file_name = os.path.join(cache_file,f'{dataset}_smooth_{side}_values.pt')
    else:
        feature_file_name = os.path.join(cache_file,f'{dataset}_rough_{side}_features.pt')
        value_file_name = os.path.join(cache_file,f'{dataset}_rough_{side}_values.pt')
    torch.save(vector,feature_file_name)
    torch.save(value,value_file_name)
            

def build_graph(args:argparse.Namespace):
    '''
    生成训练集的邻接表, 邻接矩阵
    '''
    df_train = pd.read_csv(args.trainFilePath)
    # df_test = pd.read_csv(args.testFilePath)
    R = torch.zeros((user_size,item_size)).to(args.device)
    for _,col,row in df_train.itertuples():
        R[col,row]=1
    
    torch.save(R,f'{cache_file}/train_rate_tensor.pkl')
    
    print('degree matrics')
    user_degree = R.sum(1).pow(-0.5)
    item_degree = R.sum(0).pow(-0.5)
    #FIXME: released 版本item 度未开根号
    
    user_degree[torch.isinf(user_degree)]=0
    item_degree[torch.isinf(item_degree)]=0
    
    print('Du,Di')
    Du = torch.diag(user_degree).to(args.device)
    Di = torch.diag(item_degree).to(args.device)
    print('Au,Ai')
    Au = Du.mm(R).mm(Di.pow(2)).mm(R.t()).mm(Du)
    Ai = Di.mm(R.t()).mm(Du.pow(2)).mm(R).mm(Di)
    
    print(f'User Side Adjancy Marix shape {Au.shape}')
    print(f'Item Side Adjancy Marix shape {Ai.shape}')
    
    del df_train,R,user_degree,item_degree,Du,Di 
    gc.collect()

    print('cal spectral feature')
    cal_spectral_feature(Au,int(smooth_ratio*user_size),'user',True)
    if rough_ratio!=0:
        cal_spectral_feature(Au,int(rough_ratio*user_size),'user',False)
    
    
    cal_spectral_feature(Ai,int(smooth_ratio*item_size),'item',True)
    if rough_ratio!=0:
        cal_spectral_feature(Ai,int(rough_ratio*item_size),'item',False)
    
    del Au,Ai 
    gc.collect()


def build_graph_sp(args):
    df = pd.read_csv(args.trainFilePath)
    df = df.drop_duplicates()
    col = df.iloc[:,0]
    row = df.iloc[:,1]
    R = sp.csr_matrix(([1]*df.shape[0],(col,row)))
    torch.save(torch.Tensor(R.todense()),f'{cache_file}/train_rate_tensor.pkl')
    
    user_degree = np.power(R.sum(1),-0.5).flatten()
    item_degree = np.power(R.sum(0),-0.5).flatten()
    user_degree[np.isinf(user_degree)]=0
    item_degree[np.isinf(item_degree)]=0
    Du = sp.diags(np.array(user_degree).flatten(),0)
    Di = sp.diags(np.array(item_degree).flatten(),0)
    Au = Du.dot(R).dot(Di.power(2)).dot(R.transpose()).dot(Du)
    Ai = Di.dot(R.transpose()).dot(Du.power(2)).dot(R).dot(Di)
    
    # Au = torch.Tensor(Au.todense())
    # Ai = torch.Tensor(Ai.todense())
    Au = Au.tocoo()
    Ai = Ai.tocoo()
    
    Au_sp=torch.sparse_coo_tensor(torch.Tensor([Au.row.tolist(),Au.col.tolist()]),
                             torch.Tensor(Au.data))
    Ai_sp=torch.sparse_coo_tensor(torch.Tensor([Ai.row.tolist(),Ai.col.tolist()]),
                             torch.Tensor(Ai.data))
    
    
    print(f'User Side Adjancy Marix shape {Au.shape}')
    print(f'Item Side Adjancy Marix shape {Ai.shape}')
    
    del df,R,user_degree,item_degree,Du,Di,Au,Ai
    gc.collect()

    print('cal user side smooth spectral feature')
    cal_spectral_feature(Au_sp,int(smooth_ratio*user_size),'user',True)
    if rough_ratio!=0:
        print('cal user side rough spectral feature')
        cal_spectral_feature(Au_sp,int(rough_ratio*user_size),'user',False)
    
    print('cal item side smooth spectral feature')
    cal_spectral_feature(Ai_sp,int(smooth_ratio*item_size),'item',True)
    if rough_ratio!=0:
        print('cal item side rough spectral feature')
        cal_spectral_feature(Ai_sp,int(rough_ratio*item_size),'item',False)
    
    del Au_sp,Ai_sp 
    gc.collect()
    
    
    
    
    


if __name__=='__main__':
    '''
    ml_100k :
        smooth ratio : 0.2
        rough ratio : 0.002
    pinterest :
        smooth ratio : 0.05
        rough ratio : 0
        节点数过多时, 哪怕使用稀疏矩阵进行加速, 节点数过多时的谱分解依然不快, 但至少0.05 的smooth 能在12 分钟跑完.
    '''
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset',type=str,default='pinterest')
    parse.add_argument('--device',type=str,default='cpu')
    parse.add_argument('--smooth-ratio',type=float,default=0.1)
    parse.add_argument('--rough-ratio',type=float,default=0)
    parse.add_argument('--sp',type = bool , default=True)
    args = parse.parse_args()
    print(args)
    
    args.trainFilePath = os.path.abspath(f'../datasets/{args.dataset}/train_sparse.csv' )
    # args.testFilePath = f'./dataset/{args.dataset}/test_sparse.csv' 
    # args.device = 'mps' if  torch.backends.mps.is_available() else 'cpu'
    dataset = args.dataset
    
    cache_file = os.path.join(cache_file,dataset)

    user_size,item_size  = checkAndGetNodeSize(dataset)
    if not  os.path.exists(cache_file):
        os.mkdir(cache_file)
    if args.sp:
        build_graph_sp(args)
    else:
        build_graph(args)
    
    




