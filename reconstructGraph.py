import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import random
from collections import Counter
from collections import defaultdict
import dgl
from dgl.data import citation_graph as citegrh

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


def load_data(data):
    data = eval("citegrh.load_"+data+"()") #load_cora()
    features = data.features
    labels = data.labels
    train_mask = data.train_mask.astype(bool)
    val_mask = data.val_mask.astype(bool)
    test_mask = data.test_mask.astype(bool)
    g = data.graph
    # add self loop
    # g.remove_edges_from(nx.selfloop_edges(g))
    g = dgl.DGLGraph(g)
    # g.add_edges(g.nodes(), g.nodes())
    train_idx = [i for i,d in enumerate(data.train_mask) if d == 1]
    val_idx = [i for i,d in enumerate(data.val_mask) if d == 1]
    test_idx = [i for i,d in enumerate(data.test_mask) if d == 1]
    lbls = np.zeros((labels.shape[0],np.max(labels)+1))
    lbls[[i for i in range(labels.shape[0])],labels] = 1
    y_train = np.zeros(lbls.shape)
    y_val = np.zeros(lbls.shape)
    y_test = np.zeros(lbls.shape)
    y_train[train_mask, :] = lbls[train_mask, :]
    y_val[val_mask, :] = lbls[val_mask, :]
    y_test[test_mask, :] = lbls[test_mask, :]
    adj = g.adjacency_matrix().to_dense().numpy()
    # return g.adjacency_matrix().numpy(), sp.coo_matrix(features.numpy()),  torch.LongTensor(train_idx), torch.LongTensor(val_idx),torch.LongTensor(test_idx)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, lbls

def statisticEdges(adj, labels):
    edges = sp.coo_matrix(adj)

    edges_intraCom = np.sum(labels[edges.row] == labels[edges.col])
    edges_interCom = edges.nnz - edges_intraCom
    return edges_intraCom, edges_interCom

def addEdges(adj, labels, edges_interCom, rateList, dataname, notisolate=None):
    # add edges according to node degree
    num_labels = len(np.unique(labels))
    # statistic node degree
    degree = np.array(np.sum(adj, axis=0)).squeeze()
    nodes_everyClass =[]

    member_class = []
    for l in np.unique(labels):
        member_class.append(np.where(labels == l)[0])
        nodes_everyClass.append(np.sum(labels == l))

    for rate in rateList:
        edges_added = int(edges_interCom * rate)
        nodes_added = np.random.choice(np.arange(adj.shape[0]), edges_added, (degree/np.sum(degree)).tolist())
        node_edge_interEdges = Counter(nodes_added)


        value_list = []
        for key, value in node_edge_interEdges.items():
            value_list.extend([key for j in range(value)])

        sourceList = []
        targetList = []
        while True:
            source = np.random.permutation(value_list)
            target = np.random.permutation(value_list)
            diffEdges = source != target
            source = source[diffEdges]
            target = target[diffEdges]

            difflabels = labels[source] != labels[target]
            sourceList.extend(source[difflabels])
            targetList.extend(target[difflabels])

            noRepeatList = []
            for i in range(len(sourceList) - 1):
                if np.sum((sourceList[(i+1):] == targetList[i]) & (targetList[(i+1):] == sourceList[i])) == 0:
                    noRepeatList.append(np.int32(i))
            sourceList = np.array(sourceList)[noRepeatList].tolist()
            targetList = np.array(targetList)[noRepeatList].tolist()



            noexistEdgesId = [i for i in range(len(sourceList)) if adj[sourceList[i], targetList[i]] == 0 ]
            sourceList = np.array(sourceList)[noexistEdgesId].tolist()
            targetList = np.array(targetList)[noexistEdgesId].tolist()

            if len(sourceList) >= edges_added: break

        source = np.array(sourceList[:edges_added])
        target = np.array(targetList[:edges_added])
        print(source[:5])
        edges = sp.coo_matrix(adj)
        row = np.hstack((edges.row, np.array(source), np.array(target)))
        col = np.hstack((edges.col, np.array(target), np.array(source)))
        data = np.ones(len(row))
        edges = sp.coo_matrix((data,(row,col)))
        
        graph = nx.to_dict_of_dicts(nx.from_numpy_matrix(edges.todense()))

        filename = 'data/' + 'ind.' +  dataname + '.' + str(rate) + '.graph'
        with open(filename, 'wb') as f:
            pkl.dump(graph, f, protocol=pkl.HIGHEST_PROTOCOL)


        # print('over')




if __name__ == '__main__':
    filename = sys.argv[1]
    rateList = [float(sys.argv[2])]
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels= load_data(filename)
    label_sum = None
    labels = np.where(labels > 0)[1]
    edges_intraCom, edges_interCom = statisticEdges(adj, labels)

    addEdges(adj, labels, edges_interCom, rateList, filename, notisolate = label_sum)

    print('process over')
