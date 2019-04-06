import os
import re
import random
import numpy as np
import networkx as nx
from time import time

# = = = = = = = = = = = = = = = 

# 'atoi' and 'natural_keys' taken from: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def random_walk(graph,node,walk_length):
    walk = [node]
    for i in range(walk_length):
        neighbors = graph.neighbors(walk[i])
        walk.append(random.choice(list(neighbors)))
    return walk

def generate_walks(graph,num_walks,walk_length):
    '''
    samples num_walks walks of length walk_length+1 from each node of graph
    '''
    graph_nodes = graph.nodes()
    n_nodes = len(graph_nodes)
    walks = []
    for i in range(num_walks):
        nodes = np.random.permutation(graph_nodes)
        for j in range(n_nodes):
            walk = random_walk(graph, nodes[j], walk_length)
            walks.append(walk)
    return walks

def random_walk_node2vec(graph,node,walk_length,shortest_path_lengths):
    p,q = 1,2 # 1,2 captures the node roles
    walk = [node]
    neighbors = graph.neighbors(node)
    walk.append(random.choice(list(neighbors)))
    for i in range(1,walk_length):
        neighbors = list(graph.neighbors(walk[i]))
        path_lengths = np.array([shortest_path_lengths[walk[i-1]][neighbor] for neighbor in neighbors])
        alphas = np.empty(len(neighbors))
        alphas[path_lengths == 0] = 1/p
        alphas[path_lengths == 1] = 1
        alphas[path_lengths == 2] = 1/q
        alphas /= np.sum(alphas)
        walk.append(np.random.choice(neighbors, p=alphas))
    return walk

def generate_walks_node2vec(graph,num_walks,walk_length):
    graph_nodes = graph.nodes()
    n_nodes = len(graph_nodes)
    walks = []
    shortest_path_lengths = nx.shortest_path_length(graph)
    shortest_path_lengths = [[source,targets] for source,targets in shortest_path_lengths]
    idx = [source for source,targets in shortest_path_lengths]
    dicts = [targets for source,targets in shortest_path_lengths]
    shortest_path_lengths = dict(zip(idx,dicts))
    for i in range(num_walks):
        nodes = np.random.permutation(graph_nodes)
        for j in range(n_nodes):
            walk = random_walk_node2vec(graph, nodes[j], walk_length, shortest_path_lengths)
            walks.append(walk)
    return walks

# = = = = = = = = = = = = = = =

pad_vec_idx = 1685894 # 0-based index of the last row of the embedding matrix (for zero-padding)

# parameters
num_walks = 5
walk_length = 6
max_doc_size = 110 # maximum number of 'sentences' (walks) in each pseudo-document

path_root = '/home/lucas/Desktop/MVA/Altegrad/Kaggle'
path_to_data = path_root + '/data/'

# = = = = = = = = = = = = = = =

def main():

    start_time = time() 

    edgelists = os.listdir(path_to_data + 'edge_lists/')
    edgelists.sort(key=natural_keys) # important to maintain alignment with the targets!

    docs = []
    for idx,edgelist in enumerate(edgelists):
        g = nx.read_edgelist(path_to_data + 'edge_lists/' + edgelist) # construct graph from edgelist
        doc = generate_walks_node2vec(g,num_walks,walk_length) # create the pseudo-document representation of the graph
        docs.append(doc)
        
        if idx % round(len(edgelists)/10) == 0:
            print(idx)

    print('documents generated')
    
    # truncation-padding at the document level, i.e., adding or removing entire 'sentences'
    docs = [d+[[pad_vec_idx]*(walk_length+1)]*(max_doc_size-len(d)) if len(d)<max_doc_size else d[:max_doc_size] for d in docs] 

    docs = np.array(docs).astype('int')
    print('document array shape:',docs.shape)

    np.save(path_to_data + 'documents.npy', docs, allow_pickle=False)

    print('documents saved')
    print('everything done in', round(time() - start_time,2))

# = = = = = = = = = = = = = = =

if __name__ == '__main__':
    main()
