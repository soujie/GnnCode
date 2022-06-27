from walker import RandomWalker
from gensim.models import Word2Vec



class DeepWalk:
    def __init__(self,graph,walk_length,num_walks,workers=2) -> None:
        
        self.graph=graph
        self.w2v_model=None
        self._embeddings={}
        self.walker=RandomWalker(graph)
        self.sentences=self.walker.simulate_walker(num_walks=num_walks,walk_length=walk_length,workers=workers,verbose=1)
        
    def train(self,embed_size=128,window_size=5,workers=3,iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter
        
        print('learning embedding vectors')
        model=Word2Vec(**kwargs)
        print('done')
        
        self.w2v_model=model
        return model 
    
    def get_embedding(self):
        if not self.w2v_model :
            return {}
        self._embeddings={}
        for w in self.graph.nodes():
            self._embeddings[w]=self.w2v_model.wv[w]
        
        return self._embeddings
            
        
        
        
        