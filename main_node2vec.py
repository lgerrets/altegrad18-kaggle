'''
Reference implementation of node2vec. 
Author: Aditya Grover
For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import os

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='data/edge_lists/',
	                    help='Input graph path')

	parser.add_argument('--dimensions', type=int, default=16,
	                    help='Number of dimensions. Default is 16.')

	parser.add_argument('--walk-length', type=int, default=11,
	                    help='Length of walk per source. Default is 11.')

	parser.add_argument('--num-walks', type=int, default=5,
	                    help='Number of walks per source. Default is 5.')

	parser.add_argument('--window-size', type=int, default=3,
                    	help='Context size for optimization. Default is 3.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=4,
	                    help='Number of parallel workers. Default is 4.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph(path,args):
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(path, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(args,walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	
	return model.wv

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	embeddings = {}
	edgelists = os.listdir(args.input)
	for idx,edgelist in enumerate(edgelists):
		nx_G = read_graph(args.input+edgelist,args)
		G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
		G.preprocess_transition_probs()
		walks = G.simulate_walks(args.num_walks, args.walk_length)
		n2v = learn_embeddings(args,walks)
		for key in n2v.vocab.keys():
			assert int(key) not in embeddings.keys(), (int(key),embeddings.keys())
			embeddings[int(key)] = n2v.word_vec(key)
		if idx % 10000 == 0:
			print(idx,len(list(embeddings.keys())))
	keys = np.array(list(embeddings.keys()))
	order = np.argsort(keys)
	keys = keys[order]
	values = np.array(list(embeddings.values()))
	values = values[order]

	emb_old = np.load('data/embeddings_base.npy')
	emb = np.concatenate([emb_old,values],axis=1)
	np.save('data/embeddings_new.npy',emb)


if __name__ == "__main__":
	args = parse_args()
	main(args)