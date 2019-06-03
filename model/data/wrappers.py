from .mnist import *
from .mnist_nonuniform import *
from .mocap_single import *
from .mocap_many import *
from .bballs import *


def load_data(data_dir,task,dt=0.1,subject_id=0,plot=False):
	if task=='mnist':
		dataset = load_mnist_data(data_dir,dt=dt,plot=plot)
	if task=='mnist_nonuniform':
		dataset = load_mnist_nonuniform_data(data_dir,dt=dt,plot=plot)
	elif task=='mocap_many':
		dataset = load_mocap_data_many_walks(data_dir,dt=dt,plot=plot)
	elif task=='mocap_single':
		dataset = load_mocap_data_single_walk(data_dir,subject_id=subject_id,dt=dt,plot=plot)
	elif task=='bballs':
		dataset = load_bball_data(data_dir,dt=dt,plot=plot)
	[N,T,D] = dataset.train.x.shape
	return dataset, N, T, D

def plot_reconstructions(task,X,Xrec,tobs,show=False,fname='reconstruction.png'):
	if task=='mnist':
		dataset = plot_mnist_recs(X,Xrec,show=show,fname=fname)
	if task=='mnist_nonuniform':
		dataset = plot_mnist_nonuniform_recs(X,Xrec,tobs,show=show,fname=fname)
	elif 'mocap' in task:
		dataset = plot_cmu_mocap_recs(X,Xrec,show=show,fname=fname)
	elif task=='bballs':
		dataset = plot_bball_recs(X,Xrec,show=show,fname=fname)