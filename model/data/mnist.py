import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from .utils import MyDataset


def load_mnist_data(data_dir,dt=0.1,plot=True):
	fullname = os.path.join(data_dir, "rot_mnist", "rot-mnist.mat")
	data = sio.loadmat(fullname)
	
	Xread = np.squeeze(data['X'])
	Yread = np.squeeze(data['Y'])

	N = np.shape(Xread)[0]
	M = N//10

	tr_idx  = np.arange(0,N-2*M)
	Xtr = Xread[tr_idx,:,:]
	Ytr = dt*np.arange(0,Xtr.shape[1],dtype=np.float32)
	Ytr = np.tile(Ytr,[Xtr.shape[0],1])

	val_idx = np.arange(N-2*M,N-M)
	Xval = Xread[val_idx,:,:]
	Yval  = dt*np.arange(0,Xval.shape[1],dtype=np.float32)
	Yval  = np.tile(Yval,[Xval.shape[0],1])
	
	test_idx   = np.arange(N-M,N)
	Xtest = Xread[test_idx,:,:]
	Ytest = dt*np.arange(0,Xtest.shape[1],dtype=np.float32)
	Ytest = np.tile(Ytest,[Xtest.shape[0],1])

	dataset = MyDataset(Xtr,Ytr,Xval,Yval,Xtest,Ytest)

	if plot:
		x,y = dataset.train.next_batch(7)
		plt.figure(1,(20,8))
		for j in range(6):
			for i in range(16):
				plt.subplot(7,20,j*20+i+1)
				plt.imshow(np.reshape(x[j,i,:],[28,28]), cmap='gray');
				plt.xticks([]); plt.yticks([])
		plt.savefig('plots/mnist/data.png')
		plt.close()
	return dataset

def plot_mnist_recs(X,Xrec,idxs=[0,1,2,3,4],show=False,fname='reconstructions.png'):
	if X.shape[0]<np.max(idxs):
		idxs = np.arange(0,X.shape[0])
	tt = X.shape[1]
	plt.figure(2,(tt,3*len(idxs)))
	for j, idx in enumerate(idxs):
		for i in range(tt):
			plt.subplot(2*len(idxs),tt,j*tt*2+i+1)
			plt.imshow(np.reshape(X[idx,i,:],[28,28]), cmap='gray');
			plt.xticks([]); plt.yticks([])
		for i in range(tt):
			plt.subplot(2*len(idxs),tt,j*tt*2+i+tt+1)
			plt.imshow(np.reshape(Xrec[idx,i,:],[28,28]), cmap='gray');
			plt.xticks([]); plt.yticks([])
	plt.savefig(fname)
	if show is False:
		plt.close()
