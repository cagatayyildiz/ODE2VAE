import os, math, time, argparse
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf
from scipy.io import savemat
sess = tf.InteractiveSession()

from model.data.wrappers import *

########### setup params, data, etc ###########
# read params
parser = argparse.ArgumentParser(description='ODE2VAE test arguments')
parser.add_argument('--data_root', type=str, default='../neurips19/data',
                help='root of the data folder')
parser.add_argument('--task', type=str, default='mocap_many',
                help='experiment to execute')
parser.add_argument('--ckpt', type=str, default='checkpoints',
                help='checkpoints file')
parser.add_argument('--subject_id', type=int, default=0,\
                help='subject ID in mocap_single experiments')
opt = parser.parse_args()
opt = vars(opt)
for key in opt.keys():
	print('{:s}: {:s}'.format(key, str(opt[key])))
locals().update(opt)

if not os.path.exists(os.path.join('plots', task)):
	os.makedirs(os.path.join('plots', task))
ext = 'test'

dataset, N, T, D = load_data(data_root, task, subject_id=subject_id, plot=True)

saver = tf.train.import_meta_graph('{:s}.meta'.format(ckpt))
saver.restore(sess,'{:s}'.format(ckpt))

numpars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print('{:d} trainable parameters'.format(numpars))
all_var_list = tf.trainable_variables()
for i in all_var_list:
	print(i)
	
# tf.global_variables_initializer().run()
tf.tables_initializer().run()
graph = tf.get_default_graph()
dt = 0.1

x = graph.get_tensor_by_name("X:0")
t = graph.get_tensor_by_name("t:0")
is_train = graph.get_tensor_by_name("is_train:0")
D = int(sess.run(graph.get_tensor_by_name("D:0")))
q = int(sess.run(graph.get_tensor_by_name("q:0")))
L = int(sess.run(graph.get_tensor_by_name("L:0")))
amort_len = int(sess.run(graph.get_tensor_by_name("amort_len:0")))

fs_op = graph.get_tensor_by_name("f/S:0")
Tss = graph.get_tensor_by_name("Tss:0")
vt_op = graph.get_tensor_by_name("vt_L:0") # T,N,L,q # "transpose_2:0"
st_op = graph.get_tensor_by_name("st_L:0") # "transpose_3:0"
s0_log_sig_op = graph.get_tensor_by_name("s0_log_sigma_sq:0")
v0_log_sig_op = graph.get_tensor_by_name("v0_log_sigma_sq:0")
Xrec_op   = graph.get_tensor_by_name("x_rec_mu_mu:0")  # T*N,D # rec_mean:0
Xrec_op_L = graph.get_tensor_by_name("x_rec_mu_L:0") # T*N*L,D 


def integrate(X,L=5,ts=None,fix_BNN=False,fix_enc=False):
    if ts is None:
        ts = dt * np.arange(0,X.shape[1],dtype=np.float32)
    feed_dict = {x:X, t:ts, is_train:False,Tss:len(ts)}
    if fix_BNN:
        feed_dict[fs_op] = np.array([-20.0],dtype=np.float32)
    if fix_enc:
        zero = -20*np.ones([X.shape[0],3])
        feed_dict[s0_log_sig_op] = zero
        feed_dict[v0_log_sig_op] = zero
    sts = []
    for l in range(L):
        st = sess.run(st_op, feed_dict=feed_dict)
        if 'nonuniform' in task:
        	st = np.expand_dims(st,2)
        sts.append(st)
    st = np.concatenate(sts,2)
    return st # T,N,L,q


def reconstruct(X,L=5,ts=None,fix_BNN=False,fix_enc=False):
    if ts is None:
        ts = dt * np.arange(0,X.shape[1],dtype=np.float32)
    feed_dict = {x:X, t:ts, is_train:False}
    if fix_BNN:
        feed_dict[fs_op] = np.array([-20.0],dtype=np.float32)
    if fix_enc:
        zero = -20*np.ones([X.shape[0],3])
        feed_dict[s0_log_sig_op] = zero
        feed_dict[v0_log_sig_op] = zero
    Xrecs = []
    for l in range(L):
        # Xrec = sess.run(Xrec_op, feed_dict={x: X, t:ts, is_train:False, Tss:X.shape[1]})
        Xrec = sess.run(Xrec_op_L, feed_dict={x: X, t:ts, is_train:False, Tss:X.shape[1]})
        Xrec = np.reshape(Xrec,(-1,X.shape[0],D)) # [T,N,D]
        Xrec = np.transpose(Xrec,[1,0,2]) # [N,T,D]
        Xrec = np.expand_dims(Xrec,1)
        Xrecs.append(Xrec)
    Xrec = np.concatenate(Xrecs,1) # [N,L,T,D]
    return Xrec


L = 5
X = dataset.test.next_batch(5)[0]
ts = dt * np.arange(0,X.shape[1],dtype=np.float32)
if 'nonuniform' in task:
	ts = np.tile(ts,[X.shape[0],1])
zt = integrate(X,L=L,ts=ts) # sampled latent trajectories
if q==2 or q==3:
	fname='plots/{:s}/latent_{:s}.png'.format(task,ext)
	cols = ['b','r','g','m','c','k','y']
	[T,N,L,q] = zt.shape
	zt = np.transpose(zt,[1,2,3,0]) # N,L,q,T
	plt.figure(1,(8,8))
	if q==3:
		from mpl_toolkits.mplot3d import axes3d
		ax = plt.subplot(111,projection='3d')
	for j in range(N):
		for i in range(L):
			if q==2:
				ax.plot(zt[j,i,0,:],zt[j,i,1,:],'-',color=cols[i])
				ax.scatter(zt[j,i,0,0],zt[j,i,1,0],'o',color='k',linewidth=1)
			else:
				ax.plot(zt[j,i,0,:],zt[j,i,1,:],zt[j,i,2,:],'-',color=cols[i])
				ax.scatter(zt[j,i,0,0],zt[j,i,1,0],zt[j,i,2,0],'o',color='k',linewidth=1)
	plt.savefig(fname)
	plt.close()


X,ts = dataset.train.next_batch(5)
if 'nonuniform' not in task:
    ts = ts[0]
Xrec = reconstruct(X,L=1,ts=ts)  # samples from the decoder
Xrec = np.squeeze(Xrec,1) # N,T,D
plot_reconstructions(task,X,Xrec,ts,show=False,fname='plots/{:s}/rec_{:s}.png'.format(task,'train'))

X = dataset.test.next_batch(5)[0]
ts = dt * np.arange(0,X.shape[1],dtype=np.int32)
if 'nonuniform' in task:
	ts = np.tile(ts,[X.shape[0],1])
Xrec = reconstruct(X,L=1,ts=ts)  # samples from the decoder
Xrec = np.squeeze(Xrec,1) # N,T,D
plot_reconstructions(task,X,Xrec,ts,show=False,fname='plots/{:s}/rec_{:s}.png'.format(task,ext))