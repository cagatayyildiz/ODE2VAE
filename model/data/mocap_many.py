
import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from .utils import MyDataset

def load_mocap_data_many_walks(data_dir,dt=0.1,plot=True):
	from scipy.io import loadmat
	fname = os.path.join(data_dir, 'mocap35.mat')
	mocap_data = loadmat(fname)

	Xtest = mocap_data['Xtest']
	Ytest = dt*np.arange(0,Xtest.shape[1],dtype=np.float32)
	Ytest = np.tile(Ytest,[Xtest.shape[0],1])
	Xval  = mocap_data['Xval']
	Yval  = dt*np.arange(0,Xval.shape[1],dtype=np.float32)
	Yval  = np.tile(Yval,[Xval.shape[0],1])
	Xtr   = mocap_data['Xtr']
	Ytr   = dt*np.arange(0,Xtr.shape[1],dtype=np.float32)
	Ytr   = np.tile(Ytr,[Xtr.shape[0],1])

	dataset = MyDataset(Xtr,Ytr,Xval,Yval,Xtest,Ytest)

	if plot:
		x,y = dataset.train.next_batch()
		plt.figure(2,(10,20))

		for i in range(50):
			plt.subplot(10,5,i+1)
			plt.title('sensor-{:d}'.format(i+1))
			for j in range(5):
				plt.plot(x[j,:,i])
		plt.tight_layout()
		plt.savefig('plots/mocap_many/data.png')
		plt.close()
	return dataset


def plot_cmu_mocap_recs(X,Xrec,idx=0,show=False,fname='reconstructions.png'):
	tt = X.shape[1]
	D = X.shape[2]
	nrows = np.ceil(D/5)
	lag = X.shape[1]-Xrec.shape[1]
	plt.figure(2,figsize=(20,40))
	for i in range(D):
		plt.subplot(nrows,5,i+1)
		plt.plot(range(0,tt),X[idx,:,i],'r.')
		plt.plot(range(lag,tt),Xrec[idx,:,i],'b.-')
	plt.savefig(fname)
	if show is False:
		plt.close()