import os
import numpy as np


para = {}
para['cora'] = [(32,0.6),(64,0.7)]
para['citeseer'] = [(16,0.8),(128,0.8)]
para['pubmed'] = [(256,0.6)]

record = {}
for dataset in ['cora','citeseer','pubmed']:
	para_record = []
	acc_record = []
	for p in para[dataset]:
		hidden = p[0]
		dropout = p[1]
		para_record.append(list(p))
		os.system('python cgcn.py --dataset {} --hidden_dim {} --dropout {}'.format(dataset,hidden,dropout))
		acc = np.load(dataset+'.npy').tolist()
		acc_record.append(acc)
		os.remove(dataset+'.npy')
	record[dataset] = [acc_record,para_record]
	print(dataset)
	print(record[dataset])
	print("\n")

np.save('record', record)
record = np.load('record.npy').item()
for k in record.keys():
	print(k)
	for i in range(len(record[k][0])):
		print("about loss_acc and acc_acc and para {} {}".format(record[k][0][i],record[k][1][i]))

	print("\n")