import os
import scipy.io as scio
import numpy as np
import torch
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import random
from collections import Counter
from collections import defaultdict
import dgl
from dgl.data import citation_graph as citegrh

class Dataload():

	def load(self,dataset,train,val):
		data = scio.loadmat(dataset)
		if type(data['N']) is np.ndarray:
			adj = data['N']
		else:
			adj = np.array(data['N'].todense())
		if type(data['X']) is np.ndarray:
			fea = data['X']
		else:
			fea = np.array(data['X'].todense())
		features = fea[:,:-2]
		max = np.max(features,axis=0)
		min = np.min(features,axis = 0)
		s = max - min
		features = (features - min) / s
		labels = fea[:,[-1]].squeeze() - 1
		n = len(labels)
		degree = np.sum(adj,axis=1)
		indices = np.argsort(degree)
		train_idx = indices[:int(n*train)]	
		val_idx = indices[int(n*train):int(n*(train+val))]
		test_idx = indices[int(n*(train+val)):]

		return torch.Tensor(adj),torch.Tensor(features),torch.LongTensor(labels),torch.LongTensor(train_idx),torch.LongTensor(val_idx),torch.LongTensor(test_idx)


	def load_data(self,dataset_str, rate=None): 
		"""Load data."""
		data = eval('citegrh.load_'+dataset_str+'()') 
		features = torch.FloatTensor(data.features)
		labels = torch.LongTensor(data.labels)
		train_mask = torch.IntTensor(data.train_mask)
		val_mask = torch.IntTensor(data.val_mask)
		test_mask = torch.IntTensor(data.test_mask)
		g = data.graph
		# add self loop
		# g.remove_edges_from(nx.selfloop_edges(g))
		g = dgl.DGLGraph(g)
		# g.add_edges(g.nodes(), g.nodes())
		train_idx = [i for i,d in enumerate(data.train_mask) if d == 1]
		val_idx = [i for i,d in enumerate(data.val_mask) if d == 1]
		test_idx = [i for i,d in enumerate(data.test_mask) if d == 1]

		if rate == None or rate == 0:
			adj = g.adjacency_matrix().to_dense()
		else:
			names = [ str(rate) + '.graph']
			objects = []			
			for i in range(len(names)):
				with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
					if sys.version_info > (3, 0):
						objects.append(pkl.load(f, encoding='latin1'))
					else:
						objects.append(pkl.load(f))
			
			graph = objects[0]
			
			adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).todense()

		return torch.Tensor(adj),torch.Tensor(features),torch.LongTensor(labels),torch.LongTensor(train_idx),torch.LongTensor(val_idx),torch.LongTensor(test_idx)
	    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
