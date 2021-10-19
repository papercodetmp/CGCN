# CGCN (Semi-supervised Node Classification with Community-drivenGraph Convolution Network)

please install dgl package before running code

dgl == 0.4.3
pytorch == 1.4.0


1. The files include cgcn.py, dataload.py. reconstructGraph.py and train.py.
  cgcn.py: CGCN model
  train.py: training model
  

2. please use

  (1). **python cgcn.py --dataset dataname --mod 0.1 --hidden_dim 32 --dropout 0.6 --lr 0.01 --weight_decay 0.0001** for running CGCN model, where dataname = {cora, citeseer, pubmed}.

  (2). You also could run python cgcn.py for dataset cora.

  (3). For varifying CGCN model on robustness, you could set the parameter **rate**(rate=0.1 for default) in cgcn.py.

  (4)、For running CGCN model on all standerd datasets(cora, citeseer, pubmed), **train.py** could be used directly.

