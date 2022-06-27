from unittest import result
import numpy as np
from joblib import Parallel,delayed

def sample(src_nodes :list,sample_num:int,neighrbor_dict:dict):
    ''' 一阶采样'''
    result=[]
    # for sid in src_nodes:
    result.append(np.random.choice(neighrbor_dict[src_nodes],sample_num))
    return np.asarray(result).flatten()

def mutilhop_sampling(src_nodes:list,sample_nums:list,neighrbor_dict:dict,pd=False):
    ''' 多阶采样'''
    sample_result=[src_nodes]
    
    if pd:
        sample_result=Parallel(4)(delayed(sample)(src_nodes[k],num,neighrbor_dict) for k,num in enumerate(sample_nums))
    else:
        for k,num in enumerate(sample_nums):
            sample_result.append(sample(src_nodes[k],num,neighrbor_dict))
    # return np.asarray(sample_result).flatten()
    return sample_result

if __name__=='__main__':
    dic={0:[1,2,3],1:[1,3],2:[1,3]}
    src_nodes=np.asarray([0,1])
    sample_nums=[1,2]
    a=mutilhop_sampling(src_nodes,sample_nums,dic)
    print(a)