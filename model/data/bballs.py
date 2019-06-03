import hickle as hkl
import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from .utils import MyDataset

def load_bball_data(data_dir,dt=0.1,plot=True):
	Xtr = hkl.load(os.path.join(data_dir, "bouncing_ball_data", "training.hkl"))
	# Xtr = Xtr[0:500,:,:]
	Ytr   = dt*np.arange(0,Xtr.shape[1],dtype=np.float32)
	Ytr   = np.tile(Ytr,[Xtr.shape[0],1])

	Xval = hkl.load(os.path.join(data_dir, "bouncing_ball_data", "val.hkl"))
	Yval  = dt*np.arange(0,Xval.shape[1],dtype=np.float32)
	Yval  = np.tile(Yval,[Xval.shape[0],1])

	Xtest = hkl.load(os.path.join(data_dir, "bouncing_ball_data", "test.hkl"))
	Ytest = dt*np.arange(0,Xtest.shape[1],dtype=np.float32)
	Ytest = np.tile(Ytest,[Xtest.shape[0],1])

	dataset = MyDataset(Xtr,Ytr,Xval,Yval,Xtest,Ytest)

	if plot:
		X,y = dataset.train.next_batch(5)
		tt = min(20,X.shape[1])
		plt.figure(2,(tt,5))
		for j in range(5):
			for i in range(tt):
				plt.subplot(5,tt,j*tt+i+1)
				plt.imshow(np.reshape(X[j,i,:],[32,32]), cmap='gray');
				plt.xticks([]); plt.yticks([])
		plt.savefig('plots/bballs/data.png')
		plt.close()
	return dataset

def plot_bball_recs(X,Xrec,idxs=[0,1,2,3,4],show=False,fname='reconstructions.png'):
	if X.shape[0]<np.max(idxs):
		idxs = np.arange(0,X.shape[0])
	tt = min(20,X.shape[1])
	plt.figure(2,(tt,3*len(idxs)))
	for j, idx in enumerate(idxs):
		for i in range(tt):
			plt.subplot(2*len(idxs),tt,j*tt*2+i+1)
			plt.imshow(np.reshape(X[idx,i,:],[32,32]), cmap='gray');
			plt.xticks([]); plt.yticks([])
			plt.subplot(2*len(idxs),tt,j*tt*2+i+tt+1)
			plt.imshow(np.reshape(Xrec[idx,i,:],[32,32]), cmap='gray');
			plt.xticks([]); plt.yticks([])
	plt.savefig(fname)
	if show is False:
		plt.close()
