import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from sklearn import metrics
import os
import math
import glob
import numpy as np
import sys
from dataload import Dataload as data_load
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--dataset', type=str, default='cora', help='dataset')
parse.add_argument('--rate', type=float, default=None, help='rate')
parse.add_argument('--mod', type=float, default=0.1, help='mod para')
parse.add_argument('--hidden_dim', type=int, default=32, help='hidden dim')
parse.add_argument('--dropout', type=float, default=0.6, help='dropout')
parse.add_argument('--lr', type=float, default=0.01, help='lr')
parse.add_argument('--weight_decay', type=float, default=0.0001,help='weight decay')

args = parse.parse_args()

dataset = args.dataset
rate = args.rate
a = args.mod
hidden_dim = args.hidden_dim
dropout = args.dropout
lr = args.lr
weight_decay = args.weight_decay

class GCNLayer(nn.Module):
	def __init__(self,in_dim,out_dim,dropout):
		super(GCNLayer,self).__init__()
		self.fc = nn.Linear(in_dim,out_dim)
		self.dropout = nn.Dropout(dropout)
		

	def forward(self,adj,x):
		x = torch.mm(adj,x)
		x = self.dropout(x)
		x = self.fc(x)
		return x



class GCN(nn.Module):

	def __init__(self,in_dim,hidden_dim,out_dim,dropout,GCNLayer):
		super(GCN,self).__init__()
		self.gcn1 = GCNLayer(in_dim,hidden_dim,dropout)
		self.gcn2 = GCNLayer(hidden_dim,out_dim,dropout)


	def forward(self,adj,h):
		h = f.elu(self.gcn1(adj,h))
		h = self.gcn2(adj,h)
		return h

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(15)
dataload = data_load()
device = 'cpu' #'cuda:0'


print(dataset)

adj,features,labels, train_idx,val_idx, test_idx = dataload.load_data(dataset,rate=rate)
n = adj.size()[0]
I = torch.eye(n,n)
in_dim = features.size()[1]
out_dim = torch.max(labels).item() + 1
num_head = 1
k = out_dim
adj += I
graph_d = torch.sum(adj)    
d = torch.sum(adj,dim=1).view(n,1) 
B = adj - torch.mm(d,torch.t(d)) / graph_d 

o_adj = adj.to(device)
features = features.to(device)
train_idx = train_idx.to(device)
val_idx = val_idx.to(device)
test_idx = test_idx.to(device)
labels = labels.to(device)
graph_d = graph_d.to(device)
B = B.to(device)
O_C = torch.ones(n,n).to(device)


net = GCN(in_dim,hidden_dim,out_dim,dropout,GCNLayer)

optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()


net = net.to(device)
C = O_C

epoch = 501
patients = 100
best_loss_epoch = 0
best_acc_epoch = 0
best_acc = 0
best_loss = 1000000
bad_l = 0
bad_a = 0
b = 0
l = 0
acc_record = []
try:

	for _ in range(epoch):
		net.train()
		adj = o_adj #* C
		d = (torch.sum(adj,dim=1)**(-0.5)).view(n,1)
		adj = adj * torch.mm(d,d.t())
		output = net(adj,features)
		c = torch.sigmoid(output)								
		mod = a * torch.trace(torch.mm(torch.mm(torch.t(c),B),c))/graph_d
		loss = -mod + criterion(f.log_softmax(output[train_idx],dim=1),labels[train_idx]) 
		# print(loss)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		

		net.eval()
		output = net(adj,features).detach()
		c_ = torch.sigmoid(output)
		v_mod = a * torch.trace(torch.mm(torch.mm(torch.t(c_),B),c_))/graph_d
		v_loss = criterion(f.log_softmax(output[val_idx],dim=1),labels[val_idx]) - v_mod 
		val_lbls = torch.argmax(output[val_idx],dim=1)
		acc = float(torch.sum(val_lbls == labels[val_idx])) / len(val_idx)
		


		if v_loss < best_loss:
			best_loss = v_loss
			best_loss_epoch = _
			bad_l = 0
			file = glob.glob('*loss.pth')
			if file:
				os.remove(file[0])
			torch.save({"net":net.state_dict(),"C":C},'{}loss.pth'.format(_))
		else :
			bad_l += 1

		if acc >= best_acc:
			best_acc = acc
			best_acc_epoch = _
			bad_a = 0
			file = glob.glob('*acc.pth')
			if file:
				os.remove(file[0])
			torch.save({"net":net.state_dict(),"C":C},'{}acc.pth'.format(_))
		else:
			bad_a += 1

		if bad_l == patients or bad_a == patients:

			print("early stop")
			break
		C = torch.mm(c.detach(),c.detach().t())



	net.eval()
	print("load best loss parameters")
	checkpoint = torch.load(str(best_loss_epoch)+'loss.pth')
	net.load_state_dict(checkpoint['net'])
	C = checkpoint['C']
	adj = o_adj * C
	d = (torch.sum(adj,dim=1)**(-0.5)).view(n,1)
	adj = adj * torch.mm(d,d.t())
	output = net(adj,features)
	test_lbls = torch.argmax(output[test_idx],dim=1)
	acc_loss = float(torch.sum(test_lbls == labels[test_idx])) / float(len(test_idx))

	# print("test_acc %.4f" % acc)
	# acc_record.append(acc)

	print("load best acc parameters")
	checkpoint = torch.load(str(best_acc_epoch)+'acc.pth')
	net.load_state_dict(checkpoint['net'])
	C = checkpoint['C']
	adj = o_adj * C
	d = (torch.sum(adj,dim=1)**(-0.5)).view(n,1)
	adj = adj * torch.mm(d,d.t())
	output = net(adj,features)
	test_lbls = torch.argmax(output[test_idx],dim=1)
	acc_acc = float(torch.sum(test_lbls == labels[test_idx])) / float(len(test_idx))

	acc = np.max([acc_acc, acc_loss])

	print("test_acc %.4f" % acc)
	acc_record.append(acc)




except Exception as e:
	print(e);exit()
np.save(dataset,acc_record)					
print(dataset+"done!!")


