import networkx as nx
from models import deepwalk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def read_node_label(filename, skip_head=False):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y

def plot_embeddings(embeddings,):
    X, Y = read_node_label('data/wiki_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()

if __name__=='__main__':
    graph = nx.read_edgelist('./data/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model=deepwalk.DeepWalk(graph,walk_length=20,num_walks=80,workers=2)
    model.train(window_size=5,iter=3)
    
    embeddings=model.get_embedding()
    
    plot_embeddings(embeddings)

