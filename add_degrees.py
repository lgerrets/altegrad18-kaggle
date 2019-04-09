import numpy as np
import networkx as nx
import os

def main():
	emb_old = np.load('data/embeddings_base.npy')

	emb_keys = np.zeros(emb_old.shape[0])
	emb_add = np.zeros((emb_old.shape[0],1))
	edgelists = os.listdir('data/edge_lists/')
	for idx,edgelist in enumerate(edgelists):
		nx_G = nx.read_edgelist('data/edge_lists/' + edgelist, nodetype=int, create_using=nx.DiGraph())
		degrees = dict(nx_G.degree(nx_G.nodes()))

		for key in degrees.keys():
			assert emb_keys[int(key)] == 0, int(key)
			assert degrees[key] > 0
			emb_add[int(key),0] = degrees[key]
			emb_keys[int(key)] = 1
		if idx % 10000 == 0:
			print(idx,np.sum(emb_keys))
	emb_add[emb_add==0] = np.mean(emb_add)
	emb_add = emb_add - np.mean(emb_add)
	emb_add = emb_add / np.std(emb_add)

	print(emb_add[:10,0])

	emb = np.concatenate([emb_old,emb_add],axis=1)
	print(emb.shape)
	np.save('data/embeddings_new.npy',emb)


if __name__ == "__main__":
	main()