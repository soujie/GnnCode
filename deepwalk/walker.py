from tabnanny import verbose
from joblib import Parallel,delayed
from utils import partition_num
from itertools import chain
import random

class RandomWalker:
    def __init__(self,G) -> None:
      
        self.G=G
    
    def deepwalk_walk(self,walk_length,start_node):
        walk=[start_node]
        
        while len(walk)<walk_length:
            cur=walk[-1]
            cur_nbrs=list(self.G.neighbors(cur))
            if len(cur_nbrs)>0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk
                
    
    def _simulate_walker(self,nodes,num_walks,walk_length):
        walks=[]
        for _ in range(num_walks):
            random.shuffle(nodes) 
            for v in nodes:
                walks.append(self.deepwalk_walk(walk_length=walk_length,start_node=v))
        return walks
    
    def simulate_walker(self,num_walks,walk_length,workers=2,verbose=0):
        G=self.G 
        
        nodes=list(G.nodes())
        
        results=Parallel(n_jobs=workers,verbose=verbose)(delayed(self._simulate_walker)(nodes,num,walk_length) for num in partition_num(num_walks,workers))
        
        walks=list(chain(*results))
        
        return walks
        
        