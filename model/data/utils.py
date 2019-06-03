import numpy as np
import scipy.stats as ss

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from tensorflow.python.ops import array_ops, gen_math_ops
from tensorflow.python.framework import ops

def plot_latent(zt,q=3,L=1,show=False,fname='latent.png'):
	ztlow = zt # N,q,T
	qlow  = q
	if q > 3:
		ztt = np.transpose(zt,(0,2,1)) # N,T,q
		dims = np.shape(ztt) 
		zttr = ztt.reshape([-1,q]) # N*T,q
		qlow = 2
		ztlowtr = get_pca_rep(zttr,qlow)[0]
		ztlowt = ztlowtr.reshape([dims[0],dims[1],-1]) # N,T,q
		ztlow = np.transpose(ztlowt,(0,2,1)) # N,q,T
	cols = ['b','r','g','m','c','k','y']
	plt.figure(1,(6,6))
	if qlow==3:
		from mpl_toolkits.mplot3d import axes3d
		ax = plt.subplot(111,projection='3d')
	N = int(zt.shape[0] / L)
	for j in range(N):
		for i in range(L):
			idx = j*L+i
			if qlow==2:
				plt.plot(ztlow[idx,0,:],ztlow[idx,1,:],'--',color=cols[j]) # traj, dim, time
				plt.plot(ztlow[idx,0,0],ztlow[idx,1,0],'o',color=cols[j],linewidth=8)
			elif qlow==3:
				ax.plot(ztlow[idx,0,:],ztlow[idx,1,:],ztlow[idx,2,:],'--',color=cols[j])
	plt.savefig(fname)
	if show is False:
		plt.close()

class MyBatch:
	def __init__(self,x,y=None):
		self.x = x # high-dim data
		self.y = y # time points
		self.N = x.shape[0]
	def next_batch(self,N=None): # draw N samples from the first M samples
		if N is None or N>self.N:
			ids = np.arange(self.N)
		else:
			ids = ss.uniform.rvs(size=N)*self.N
		ids = [int(i) for i in ids]
		xs = self.x[ids,:]
		if self.y is None:
			ys = None
		else:
			ys = self.y[ids,:]
		return xs, ys

class MyDataset:
	def __init__(self,xtr,ytr,xval=None,yval=None,xtest=None,ytest=None):
		self.train = MyBatch(xtr, ytr)
		if xval is not None:
			self.val = MyBatch(xval, yval)
		if xtest is not None:
			self.test = MyBatch(xtest, ytest)


def get_pca_rep(Y,q):
	# Y - data stored in rows
	means = np.mean(Y,0)
	Yc = Y - means
	S = np.cov(Yc,rowvar=False)
	v, u = np.linalg.eig(S)
	idx = v.argsort()[::-1]
	v = np.real(v[idx])
	u = np.real(u[:,idx])
	v[v<0] = 1e-8
	Ypca = np.matmul(np.matmul(Yc,u[:,0:q]),np.diag(1/np.sqrt(v[0:q])))
	return Ypca, means, u, v


def read_amcs(fnames,q=3,crop=True):
	Ys = []
	Yspca = []
	Nfile = len(fnames)
	initY = np.zeros((Nfile,50))
	for i in range(Nfile):
		Y_, initY_ = read_amc(fnames[i],crop)
		Ys.append(Y_)
		initY[i,:] = initY_
	Nrows = [Y.shape[0] for Y in Ys]
	data = np.zeros((0,Ys[0].shape[1]))
	for i in range(Nfile):
		data = np.concatenate((data,Ys[i]))
	means = np.mean(data,0)
	datac = data - means
	S = np.cov(datac,rowvar=False)
	v, u = np.linalg.eig(S)
	idx = v.argsort()[::-1]
	v = np.real(v[idx])
	u = np.real(u[:,idx])
	v[v<0] = 1e-8
	pca_data = np.matmul(np.matmul(datac,u[:,0:q]),np.diag(1/np.sqrt(v[0:q])))
	j = 0
	for i in range(Nfile):
		Ys[i] = datac[j:j+Nrows[i],:]
		Yspca.append(pca_data[j:j+Nrows[i],:])
		j += Nrows[i]
	return Ys,Yspca,means,v,u,initY


def read_amc(fname,crop=False,onset=0):
	with open(fname) as f:
		content = f.readlines()
	content = [x.strip() for x in content]
	while content[0] is not '1':
		content = content[1:]
	L = 30 # number of lines for each observed time point
	N = int(len(content)/L)
	D = 62
	data = np.zeros((N,D))
	for i in range(N):
		arr = [content[i*L+j].split()[1:] for j in range(1,L)]
		flat_arr = [float(item) for subarr in arr for item in subarr]
		data[i,:] = flat_arr
	if onset>0:
		data = data[onset:,:]
	initY = data[0,:]
	if crop:
		idx_ = [a for subarr in [np.arange(0,31),np.arange(36,43),np.arange(48,54),np.arange(55,61)] for a in subarr]
		data = data[:,idx_]
		# init Y
		initY = data[0,:]
		# data difference
		data[:-1,0:3]=data[1:,0:3]-data[:-1,0:3]; data[-1,0:3]=data[-2,0:3]

	return data,initY


def save_amc(D,fname='pred.amc'):
	if D.shape[1] != 62:
		raise ValueError('Input matrix does not have 62 dimensions.')
	with open(fname,'w') as f:
		f.write('#!Python matrix to amc conversion\n');
		f.write(':FULLY-SPECIFIED\n');
		f.write(':DEGREES\n');
		for frame in range(D.shape[0]):
			f.write('{:d}\n'.format(frame+1))
			f.write('root {:f} {:f} {:f} {:f} {:f} {:f}\n'.format(*list(D[frame,0:6])))
			f.write('lowerback {:f} {:f} {:f}\n'.format(*list(D[frame,6:9])))
			f.write('upperback {:f} {:f} {:f}\n'.format(*list(D[frame,9:12])))
			f.write('thorax {:f} {:f} {:f}\n'.format(*list(D[frame,12:15])))
			f.write('lowerneck {:f} {:f} {:f}\n'.format(*list(D[frame,15:18])))
			f.write('upperneck {:f} {:f} {:f}\n'.format(*list(D[frame,18:21])))
			f.write('head {:f} {:f} {:f}\n'.format(*list(D[frame,20:23])))
			f.write('rclavicle {:f} {:f}\n'.format(*list(D[frame,24:26])))
			f.write('rhumerus {:f} {:f} {:f}\n'.format(*list(D[frame,26:29])))
			f.write('rradius {:f}\n'.format((D[frame,29])))
			f.write('rwrist {:f}\n'.format((D[frame,30])))
			f.write('rhand {:f} {:f}\n'.format(*list(D[frame,31:33])))
			f.write('rfingers {:f}\n'.format((D[frame,33])))
			f.write('rthumb {:f} {:f}\n'.format(*list(D[frame,34:36])))
			f.write('lclavicle {:f} {:f}\n'.format(*list(D[frame,36:38])))
			f.write('lhumerus {:f} {:f} {:f}\n'.format(*list(D[frame,38:41])))
			f.write('lradius {:f}\n'.format((D[frame,41])))
			f.write('lwrist {:f}\n'.format((D[frame,42])))
			f.write('lhand {:f} {:f}\n'.format(*list(D[frame,43:45])))
			f.write('lfingers {:f}\n'.format((D[frame,45])))
			f.write('lthumb {:f} {:f}\n'.format(*list(D[frame,46:48])))
			f.write('rfemur {:f} {:f} {:f}\n'.format(*list(D[frame,48:51])))
			f.write('rtibia {:f}\n'.format((D[frame,51])))
			f.write('rfoot {:f} {:f}\n'.format(*list(D[frame,52:54])))
			f.write('rtoes {:f}\n'.format((D[frame,54])))
			f.write('lfemur {:f} {:f} {:f}\n'.format(*list(D[frame,55:58])))
			f.write('ltibia {:f}\n'.format((D[frame,58])))
			f.write('lfoot {:f} {:f}\n'.format(*list(D[frame,59:61])))
			f.write('ltoes {:f}\n'.format((D[frame,61])))