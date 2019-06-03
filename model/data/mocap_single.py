
import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from .utils import MyDataset

def load_mocap_data_single_walk(data_dir,subject_id,dt=0.1,plot=True):
	fname = os.path.join(data_dir, 'mocap43.mat')
	mocap_data = loadmat(fname)

	Xtest = mocap_data['Ys'][subject_id][0]
	Xtest = np.expand_dims(Xtest,0)
	Ytest = dt*np.arange(0,Xtest.shape[1],dtype=np.float32)
	Ytest = np.tile(Ytest,[Xtest.shape[0],1])
	Xval  = mocap_data['Yobss'][subject_id][0]
	Xval  = np.expand_dims(Xval,0)
	Yval  = dt*np.arange(0,Xval.shape[1],dtype=np.float32)
	Yval  = np.tile(Yval,[Xval.shape[0],1])
	N = Xval.shape[1]
	Xtr   = Xval[:,:4*N//5,:]
	Ytr   = dt*np.arange(0,Xtr.shape[1],dtype=np.float32)
	Ytr   = np.tile(Ytr,[Xtr.shape[0],1])

	dataset = MyDataset(Xtr,Ytr,Xval,Yval,Xtest,Ytest)

	if plot:
		x,y = dataset.train.next_batch()
		plt.figure(2,(10,20))
		for i in range(50):
			plt.subplot(10,5,i+1)
			plt.title('sensor-{:d}'.format(i+1))
			plt.plot(x[0,:,i])
		plt.tight_layout()
		plt.savefig('plots/mocap_single/data.png')
		plt.close()
	return dataset
