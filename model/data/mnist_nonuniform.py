import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from .utils import MyDataset


def load_mnist_nonuniform_data(data_dir,dt=0.1,plot=True):
	fullname = os.path.join(data_dir, "rot_mnist", "rot-mnist-3s.mat")
	data = sio.loadmat(os.path.join(data_dir, "rot_mnist", 'rot-mnist-3s.mat'))
	Xtr = np.squeeze(data['X'])
	Ytr = np.squeeze(data['Y'])
	Xtr = np.flip(Xtr,1)
	idx = np.arange(Xtr.shape[0])
	np.random.shuffle(idx)
	Xtr = Xtr[idx,:,:]

	Ntr = 360
	Xval = Xtr[Ntr:Ntr+Ntr//10]
	Xtest = Xtr[0:Ntr]

	removed_angle = 3
	[N,T,D] = Xtest.shape
	num_gaps = 5
	ttr = np.zeros([N,T-num_gaps])
	Xtr = np.zeros([N,T-num_gaps,D])
	for i in range(N):
		idx = np.arange(0,T)
		d = {removed_angle}
		while len(d) < num_gaps:
			d.add(np.random.randint(0,T))
		idx = np.delete(idx,list(d))
		Xtr[i,:,:] = Xtest[i,idx,:]
		ttr[i,:] = dt*idx

	ttest = dt*np.tile(np.arange(0,T).reshape((1,-1)),[N,1])
	tval  = dt*np.tile(np.arange(0,T).reshape((1,-1)),[Xval.shape[0],1])

	dataset = MyDataset(Xtr,ttr,Xval,tval,Xtest,ttest)
	if plot:
		N = 5
		x,ts = dataset.train.next_batch(N)
		fig,ax = plt.subplots(N,16)
		for ax_ in ax:
			for ax__ in ax_:
				ax__.set_xticks([]);ax__.set_yticks([])
		plt.axis('off')
		fig.set_size_inches(2*N, 2*N)
		for j in range(N):
			ts_ = [int(10*i) for i in ts[j]] # 0.1,0.2,... ---> 1,2,...
			for i,t in enumerate(ts_):
				ax[j,t].imshow(np.reshape(x[j,i,:],[28,28]), cmap='gray')
		plt.savefig('plots/mnist_nonuniform/data.png')
		plt.close()
	return dataset


def plot_mnist_nonuniform_recs(X,Xrec,tres,show=False,fname='reconstructions.png'):
	N = X.shape[0]
	fig,ax = plt.subplots(2*N,16)
	for ax_ in ax:
		for ax__ in ax_:
			ax__.set_xticks([]);ax__.set_yticks([])
	plt.axis('off')
	fig.set_size_inches(3*N, 3*N)
	for j in range(N):
		ts = [int(10*i) for i in tres[j]] # 0.1,0.2,... ---> 1,2,...
		for i,t in enumerate(ts):
			ax[2*j,t].imshow(np.reshape(X[j,i,:],[28,28]), cmap='gray')
			ax[2*j+1,t].imshow(np.reshape(Xrec[j,i,:],[28,28]), cmap='gray')
	plt.savefig(fname)
	if show is False:
		plt.close()
