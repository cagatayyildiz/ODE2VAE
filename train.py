import os, math, time

import numpy as np
import tensorflow as tf
from scipy.io import savemat
sess = tf.InteractiveSession()

from model.data.wrappers import *
from model.data.utils import plot_latent
from model.ode2vae_args import ODE2VAE_Args


########### setup params, data, etc ###########
# read params
opt = ODE2VAE_Args().parse()
for key in opt.keys():
	print('{:s}: {:s}'.format(key, str(opt[key])))
locals().update(opt)
if not os.path.exists(os.path.join(ckpt_dir, task)):
	os.makedirs(os.path.join(ckpt_dir, task))
if not os.path.exists(os.path.join('plots', task)):
	os.makedirs(os.path.join('plots', task))
# dataset
dataset, N, T, D = load_data(data_root, task, subject_id=subject_id, plot=True)
# artificial time points
dt = 0.1
t = dt*np.arange(0,T,dtype=np.float32)
# file extensions 
if task=='bballs' or 'mnist' in task:
	ext = '{:s}_q{:d}_inst{:d}_fopt{:d}_enc{:d}_dec{:d}'.format(task,q,inst_enc_KL,f_opt,NF_enc,NF_dec)
elif 'mocap' in task:
	ext = '{:s}_q{:d}_inst{:d}_fopt{:d}_He{:d}_Hf{:d}_Hd{:d}'.format(task,q,inst_enc_KL,f_opt,He,Hf,Hd)
print('file extensions are {:s}'.format(ext))


########### training related stuff ###########
xval_batch_size = int(batch_size/2)
min_val_lhood = -1e15

xbspl = tf.placeholder(tf.int64, name="tr_batch_size")
xfpl  = tf.placeholder(tf.float32, [None,None,D], name="tr_features")
xtpl  = tf.placeholder(tf.float32, [None,None], name="tr_timepoints")

def data_map(X, y, W=T, p=0, dt=dt):
	W  += tf.random_uniform([1], 0, 1, tf.int32)[0] # needed for t to be of dim. None
	W  = tf.cast(W,tf.int32)
	rng_ = tf.range(0,W)
	t_ = tf.to_float(dt) * tf.cast(rng_,tf.float32)
	X = tf.gather(X,rng_,axis=1)
	y = tf.gather(y,rng_,axis=1)
	return X,y,t_

xtr_dataset  = tf.data.Dataset.from_tensor_slices((xfpl, xtpl)).batch(xbspl).map(data_map,8).prefetch(2)
xval_dataset = tf.data.Dataset.from_tensor_slices((xfpl, xtpl)).batch(xbspl).map(data_map,8).repeat()

xiter_ = tf.data.Iterator.from_structure(xtr_dataset.output_types, xtr_dataset.output_shapes)
if 'nonuniform' not in task:
	X, _, t = xiter_.get_next()
else:
	X, t, _ = xiter_.get_next()
xtr_init_op   = xiter_.make_initializer(xtr_dataset)
xval_init_op  = xiter_.make_initializer(xval_dataset)


########### model ###########
if 'nonuniform' not in task:
	from model.ode2vae import ODE2VAE 
else:
	from model.ode2vae_nonuniform import ODE2VAE 
vae = ODE2VAE(sess, f_opt, q, D, X, t, NF_enc=NF_enc, NF_dec=NF_dec, KW_enc=KW_enc, KW_dec=KW_dec, Nf=Nf, Ne=Ne, Nd=Nd,\
				 task=task, eta=eta, L=1, Hf=Hf, He=He, Hd=Hd, activation_fn=activation_fn, inst_enc_KL=inst_enc_KL, \
				 amort_len=amort_len, gamma=gamma)


########### training loop ###########
t0 = time.time()

print('{:>15s}'.format("epoch")+'{:>15s}'.format("total_cost")+'{:>15s}'.format("E[p(x|z)]")+'{:>15s}'.format("E[p(z)]")+'{:>15s}'.format("E[q(z)]")+\
	  '{:>16s}'.format("E[KL[ode||enc]]")+'{:>15s}'.format("valid_cost")+'{:>15s}'.format("valid_error"))
print('{:>15s}'.format("should")+'{:>15s}'.format("decrease")+'{:>15s}'.format("increase")+'{:>15s}'.format("increase")+'{:>15s}'.format("decrease")+\
	  '{:>16s}'.format("decrease")+'{:>15s}'.format("decrease")+'{:>15s}'.format("decrease"))
for epoch in range(num_epoch):
	values = [0,0,0,0,0]
	num_iter = 0
	Tss = max(min(T, T//5+epoch//2), vae.amort_len+1)
	sess.run(xtr_init_op, feed_dict = {xfpl:dataset.train.x,  xtpl:dataset.train.y,  xbspl:batch_size})
	while True:
		try:
			if np.mod(num_iter,20)==0:
				print(num_iter)
			ops_ = [vae.vae_optimizer, vae.vae_loss, vae.reconstr_lhood, vae.log_p, vae.log_q, vae.inst_enc_loss]
			values_batch = sess.run(ops_,feed_dict={vae.train:True, vae.Tss:Tss})
			values = [values[i]+values_batch[i+1] for i in range(5)]
			num_iter += 1
		except tf.errors.OutOfRangeError:
			break
	values = [values[i]/num_iter for i in range(5)]
	xtr_dataset = xtr_dataset.shuffle(buffer_size=dataset.train.N)
	sess.run(xval_init_op, feed_dict = {xfpl:dataset.val.x, xtpl:dataset.val.y, xbspl:xval_batch_size})
	val_lhood = 0 
	num_val_iter = 10
	for _ in range(num_val_iter):
		try:
			val_lhood += sess.run(vae.mean_reconstr_lhood,feed_dict={vae.train:False, vae.Tss:Tss})
		except tf.errors.OutOfRangeError:
			break
	val_lhood = val_lhood / num_val_iter / Tss
	xval_dataset = xval_dataset.shuffle(buffer_size=dataset.val.N)

	if val_lhood>min_val_lhood:
		min_val_lhood = val_lhood
		vae.save_model(ckpt_dir,ext)
		X,ttr = dataset.train.next_batch(5)
		Xrec = vae.reconstruct(X,ttr)
		zt   = vae.integrate(X,ttr)
		plot_reconstructions(task,X,Xrec,ttr,show=False,fname='plots/{:s}/rec_tr_{:s}.png'.format(task,ext))
		plot_latent(zt,vae.q,vae.L,show=False,fname='plots/{:s}/latent_tr_{:s}.png'.format(task,ext))
		X,tval = dataset.val.next_batch(5)
		Xrec = vae.reconstruct(X,tval)
		# zt   = vae.integrate(X)
		plot_reconstructions(task,X,Xrec,tval,show=False,fname='plots/{:s}/rec_val_{:s}.png'.format(task,ext))
		# plot_latent(zt,vae.q,vae.L,show=False,fname='plots/{:s}/latent_val_{:s}.png'.format(task,ext))
		val_err = -1
		if 'mnist' in task:
			X1 = X[:,amort_len:,:]
			X2 = Xrec[:,amort_len:,:]
			val_err = np.mean((X1-X2)**2)
		elif task=='bballs':
			X1 = X[:,amort_len:amort_len+10,:]
			X2 = Xrec[:,amort_len:amort_len+10,:]
			val_err = np.sum((X1-X2)**2,2)
			val_err = np.mean(val_err)
		elif task == 'mocap_single':
			diff = X[0,:,:] - Xrec[0,:,:]
			diff = diff[4*diff.shape[0]//5:,:]**2
			val_err = np.mean(diff)
		elif task == 'mocap_many':
			val_err = np.mean((X-Xrec)**2)
		print('{:>15d}'.format(epoch)+'{:>15.1f}'.format(values[0])+ '{:>15.1f}'.format(values[1])+'{:>15.1f}'.format(values[2])+\
			'{:>15.1f}'.format(-values[3])+'{:>15.1f}'.format(values[4])+'{:>15.1f}'.format(val_lhood)+'{:>15.3f}'.format(val_err))
	else:
		print('{:>15d}'.format(epoch)+'{:>15.1f}'.format(values[0])+ '{:>15.1f}'.format(values[1])+'{:>15.1f}'.format(values[2])+\
			'{:>15.1f}'.format(-values[3])+'{:>15.1f}'.format(values[4])+'{:>15.1f}'.format(val_lhood))

	if math.isnan(values[0]):
		print('*** average cost is nan. terminating...')
		break

t1 = time.time()
print('elapsed time: {:.2f}'.format(t1-t0))
