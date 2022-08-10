import pandas as pd 
import torch
import numpy as np 
def checkAndGetNodeSize(target):
    if target == 'ml_100k':
        return 943 , 1682
    elif target == 'pinterest':
        return  37501,9831
    else:
        raise ValueError('dateset must be ml_100k or pinterest')

def convert2tensor(dataset:str,task:str,format:str = 'tensor'):
    user_size, item_size = checkAndGetNodeSize(dataset)
    df = pd.read_csv(f'../datasets/{dataset}/{task}_sparse.csv')
    assert format in ['tensor','adjlist'], f'{task} only support return tensor or adjancy list'
    if format == 'tensor':
        rating = torch.zeros((user_size,item_size))
        for _,row,col in df.itertuples():
            rating[row,col] = 1
        return rating
    else:
        tmp =[ [] for _ in range(user_size)]
        for _,row,col in df.itertuples():
            tmp[row].append(col)
        return tmp 

def cal_idcg(k=20):
    idcg_set =[0]
    scores= .0
    for i in range(1,k+1):
        scores+=1/np.log2(1+i)
        idcg_set.append(scores)
    return idcg_set

def cal_score(topn_predict,gts):
    '''
    topn_pred: list , 预测结果的topn item index
    gts : 实际购买item 的index
    '''
    dcg10,dcg20,hit10,hit20 =.0,.0,.0,.0
    for k in range(len(topn_predict)):
        item = topn_predict[k]
        if gts.count(item) !=0:
            if k<=10:
                dcg10+=1/np.log2(2+k)
                hit10+=1
            dcg20+=1/np.log2(2+k)
            hit20+=1
    return dcg10,dcg20,hit10,hit20