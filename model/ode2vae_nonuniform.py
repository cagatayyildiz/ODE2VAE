import warnings
warnings.filterwarnings("ignore")

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf
import scipy.stats as ss

import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops

from .tf_utils import *

class ODE2VAE(object):
	def __init__(self, sess, f_opt, q, D, X, t, task='cmu', eta=0.001, L=1, amort_len=1,\
				Hf=100, He=20, Hd=20, Nf=2, Ne=2, Nd=2, activation_fn=tf.nn.relu, inst_enc_KL=True,\
				NF_enc=16, NF_dec=16, KW_enc=5, KW_dec=5, gamma=1.0):
		"""
		sess       - TensorFlow session object
		f_opt      - wheter drift is GP(0), NN(1), var-NN(2)
		q          - latent dim
		D          - data dimensionality
		K          - dimensionality of static latent features
		task    - network architecture - 'cmu', 'mnist', 'bballs', 'face', 'h36m', 'dog'
		eta        - learning rate
		L          - number of z0 particles (random draws) per input data point
		amort_len    - the number of data points from time 0 used in encoding
		NF_enc     - number of filters in encoder cnn
		NF_dec     - number of filters in decoder cnn
		KW_enc     - kernel width in encoder cnn
		KW_dec     - kernel width in encoder cnn
		beta       - the constant in front of KL term (beta-VAE term)
		lambda     - the weight of GAN loss
		"""
		self.sess = sess
		self.f_opt = f_opt
		self.q = q
		self.D = D
		self.L = 1
		self.Hf = Hf
		self.He = Hf
		self.Hd = Hf
		self.Nf = Nf
		self.Ne = Nf
		self.Nd = Nf
		self.activation_fn = activation_fn
		self.inst_enc_KL = inst_enc_KL
		self.task = task
		self.eta = eta
		self.amort_len = 1
		self.Tss = tf.placeholder(tf.int64, name="Tss")
		self.NF_enc = NF_enc
		self.NF_dec = NF_dec
		self.KW_enc = KW_enc
		self.KW_dec = KW_dec
		self.gamma = gamma
		print('gamma is {:.6f}'.format(self.gamma))
		self.beta = self.q / (Hf*(3*q+Nf*Hf+Nf+1)+q)
		print('beta is {:.6f}'.format(self.beta))
		self.train = tf.placeholder(tf.bool, name="is_train") # needed for batch normalization
		# self.x = tf.placeholder(tf.float32, [None, None, D], name="X") # input sequences [N,T,D]
		# self.t = tf.placeholder(tf.float32, [None, None], name="t") # [N,T]
		self.x = tf.identity(X, name="X")
		self.t = tf.identity(t, name="t")
		self.qt = tf.constant(self.q, dtype=tf.float32, name='q') # needed while loading a saved model
		self.Dt = tf.constant(self.D, dtype=tf.float32, name='D') # needed while loading a saved model
		self.Lt = tf.constant(self.L, dtype=tf.float32, name='L') # needed while loading a saved model
		self.amort_lent = tf.constant(self.amort_len, dtype=tf.int32, name='amort_len') # needed while loading a saved model
		self.create_network()
		self.create_loss()
		self.create_optimizers()
		self.sess.run(tf.global_variables_initializer())

	def set_configs(self):
		# network output types, needed for reconstruction likelihood
		if self.task=='bballs' or 'mnist' in self.task:
			self.dec_out = 'bernoulli'
		else:
			self.dec_out = 'normal'
		print(self.dec_out)
		# batch normalization objects used in static&dynamic encoders and decoder
		self.s_bns = []
		self.v_bns = []
		self.d_bns = []
		for i in range(8):
			self.s_bns.append(batch_norm(name='s_bns{:d}'.format(i)))
			self.v_bns.append(batch_norm(name='v_bns{:d}'.format(i)))
			self.d_bns.append(batch_norm(name='d_bns{:d}'.format(i)))
		if self.f_opt==0:
			Z0 = ss.uniform.rvs(size=[M,q])*4-2
			U0 = ss.norm.rvs(size=Z0.shape)*0.1
			U  = tf.Variable(U0, dtype=tf.float32, trainable=True, name='U')
			Z  = tf.Variable(Z0, dtype=tf.float32, trainable=False, name='Z')
			kern = OperatorKernel(ell0=ell0, train_ell=False)
			def f(X,t=[0]):
				with tf.variable_scope('f', reuse=tf.AUTO_REUSE):
					Kzx = kern.K(Z, X)
					f = tf.matmul(Kzx, U, transpose_a=True)
					return f
			self.f = f
			self.fmean = fmean
		elif self.f_opt==1:
			def f(X,t=[0]):
				with tf.variable_scope('f', reuse=tf.AUTO_REUSE):
					x = tf.reshape(X,[-1,2*self.q])
					for i in range(0,self.Nf+1):
						x = tf.contrib.layers.fully_connected(x, self.Hf, activation_fn=self.activation_fn)
					x = tf.contrib.layers.fully_connected(x, self.q, activation_fn=None)
					return x
			self.f = f
			self.fmean = f
		elif self.f_opt==2:
			with tf.variable_scope('f', reuse=tf.AUTO_REUSE):
				self.loglambdasq = tf.Variable(np.log(np.array([1e-4])),dtype=tf.float32,name='S')
				tf.summary.scalar('S', tf.exp(self.loglambdasq[0]))
				Ns = [2*self.q]
				for tmp in range(self.Nf):
					Ns.append(self.Hf)
				Ns.append(self.q)
				# q,H,H,H,q
				self.f_weight_means = [] # weight matrix means
				self.f_bias_means   = [] # bias vector means
				for i in range(0,self.Nf+1):
					w_ = tf.get_variable("w{:d}".format(i), shape=[Ns[i],Ns[i+1]], initializer=tf.contrib.layers.xavier_initializer())
					self.f_weight_means.append(w_)
					b_ = tf.get_variable("b{:d}".format(i), shape=[Ns[i+1]], initializer=tf.zeros_initializer())
					self.f_bias_means.append(b_)
				Ws = [] # TF weight samples
				Bs = [] # TF bias samples
				for i in range(0,self.Nf+1):
					Wi = tf.random_normal([Ns[i],Ns[i+1]])*tf.sqrt(tf.exp(self.loglambdasq)) + self.f_weight_means[i]
					Ws.append(Wi)
					Bi = tf.random_normal([Ns[i+1]])*tf.sqrt(tf.exp(self.loglambdasq)) + self.f_bias_means[i]
					Bs.append(Bi)
			def f(X,t=[0]):
				with tf.variable_scope('f', reuse=tf.AUTO_REUSE):
					x = tf.reshape(X,[-1,2*self.q])
					for i in range(0,self.Nf+1):
						if i<self.Nf:
							x = tf.nn.relu(tf.matmul(x,Ws[i])+Bs[i])
						elif i==self.Nf:
							x = tf.matmul(x,Ws[i])+Bs[i]
					return x
			def fmean(X,t=[0]):
				with tf.variable_scope('f', reuse=tf.AUTO_REUSE):
					x = tf.reshape(X,[-1,2*self.q])
					for i in range(0,self.Nf+1):
						if i<self.Nf:
							x = tf.nn.relu(tf.matmul(x,self.f_weight_means[i])+self.f_bias_means[i])
						elif i==self.Nf:
							x = tf.matmul(x,self.f_weight_means[i])+self.f_bias_means[i]
					return x
			self.f = f
			self.fmean = fmean
		# extract data points v0&s0 are computed conditioned on
		v0 = tf.gather(self.x,[i for i in range(self.amort_len)],axis=1) # N,amort_len,D
		s0 = tf.squeeze(tf.gather(self.x,[0],axis=1),axis=1) # N,D
		# extract the data points for instant encoding
		x0vs = tf.stack([self.x[:,i:-self.amort_len+i] for i in range(self.amort_len)],0) # amort_len, N, T-amort_len, D
		x0vs = tf.transpose(x0vs,[2,1,0,3]) # T-amort_len, N, amort_len, D
		x0vs = tf.reshape(x0vs,[-1,self.amort_len,self.D]) # (T-amort_len)*N, amort_len, D
		x0ss = self.x[:,:-self.amort_len,:] # N, T-amort_len, D
		x0ss = tf.transpose(x0ss,[1,0,2])   # T-amort_len, N, D
		x0ss = tf.reshape(x0ss,[-1,self.D]) # (T-amort_len)*N, D
		# static/dynamic encoders and decoder
		if 'cmu' in self.task or 'h36m' in self.task:
			s_enc_fnc = self.enc_dense
			dec_fnc   = self.dec_dense
			v_enc_fnc = self.enc_dense # enc_dense, enc_rnn
			v0 = tf.reshape(v0,[-1,self.amort_len*self.D])
			x0vs = tf.reshape(x0vs,[-1,self.amort_len*self.D]) # (T-amort_len)*N, amort_len*D
		elif 'mnist' in self.task or self.task=='bballs':
			dec_fnc = self.dec_mnist_bball
			v_enc_fnc = self.enc_mnist_bball
			s_enc_fnc = self.enc_mnist_bball
			def channel_stack(x,T):
				dim = int(np.sqrt(self.D))
				x = tf.reshape(x, shape=[-1, T, dim, dim, 1])
				x = tf.transpose(x,[0,2,3,1,4])
				x = tf.reshape(x,[-1,dim,dim,T])
				return x
			v0 = channel_stack(v0,self.amort_len) # N,sqrtD,sqrtD,amort_len
			s0 = channel_stack(s0,1) # N,sqrtD,sqrtD,1
			x0vs = channel_stack(x0vs,self.amort_len) # (T-a_len)*N,D,D,amort_len
			x0ss = channel_stack(x0ss,1) # (T-a_len)*N,D,D,1
		elif self.task=='face':
			def channel_stack(x,T):
				dim = 128
				x = tf.reshape(x, shape=[-1, T, dim, dim, 3])
				x = tf.transpose(x,[0,2,3,1,4])
				x = tf.reshape(x,[-1,dim,dim,3*T])
				return x
			v_enc_fnc = self.enc_face
			s_enc_fnc = self.enc_face
			v0  = channel_stack(v0,self.amort_len)
			s0  = channel_stack(s0,1)
			# v0 = tf.reshape(v0, shape=[-1, 128, 128, 3])
			# s0 = tf.reshape(s0, shape=[-1, 128, 128, 3])
			dec_fnc = self.dec_face
		elif self.task=='dog':
			v_enc_fnc = self.enc_dog
			s_enc_fnc = self.enc_dog
			dec_fnc = self.dec_dog
			s0 = tf.reshape(s0, shape=[-1, 640, 360, 3])
			s0 = tf.image.resize_image_with_crop_or_pad(s0,640,384)
			s0 = tf.transpose(s0,[1,2,0,3])
			s0 = tf.reshape(s0, shape=[1, 640, 384, 3])
			v0 = tf.reshape(v0, shape=[-1, 640, 360, 3])
			v0 = tf.image.resize_image_with_crop_or_pad(v0,640,384)
			v0 = tf.transpose(v0,[1,2,0,3])
			v0 = tf.reshape(v0, shape=[1, 640, 384, 3*self.amort_len])
		return v0, s0, x0vs, x0ss, v_enc_fnc, s_enc_fnc, dec_fnc

	def create_network(self):
		self.x = self.x[:,:self.Tss,:]
		self.t = self.t[:,:self.Tss]
		x0_v, x0_s, x0vs_enc, x0ss_enc, v_enc_fnc, s_enc_fnc, dec_fnc = self.set_configs() # N,sqrtD,sqrtD,amort_len
		v0_mu, v0_log_sigma_sq = v_enc_fnc(x0_v, self.q, self.v_bns, 'enc_v0') # N,q
		s0_mu, s0_log_sigma_sq = s_enc_fnc(x0_s, self.q, self.s_bns, 'enc_s0') # N,q
		v0_log_sigma_sq = tf.identity(v0_log_sigma_sq,name='v0_log_sigma_sq')
		s0_log_sigma_sq = tf.identity(s0_log_sigma_sq,name='s0_log_sigma_sq')
		mvn = tfd.MultivariateNormalDiag(loc=tf.zeros_like(v0_mu),scale_diag=tf.ones_like(v0_mu))
		eps1 = tf.random_normal(tf.shape(v0_mu), 0, 1, dtype=tf.float32)
		v0 = tf.add(v0_mu, tf.multiply(tf.sqrt(tf.exp(v0_log_sigma_sq)), eps1))
		eps2 = tf.random_normal(tf.shape(s0_mu), 0, 1, dtype=tf.float32)
		s0 = tf.add(s0_mu, tf.multiply(tf.sqrt(tf.exp(s0_log_sigma_sq)), eps2))
		q0 = mvn.log_prob(eps1) + mvn.log_prob(eps2)
		q0 = tf.expand_dims(q0,1) # N,1
		tdense = tf.identity(0.02*tf.range(0.0,401.0,delta=1.0,dtype=tf.float32),name='tdense')
		# tdense = tf.concat([tdense,tf.reshape(self.t,[-1])],0)
		vt,st,logpt = self.ode_density(v0, s0, q0, self.f, tdense) # T,N,q
		vt_mu,st_mu = self.ode(v0_mu, s0_mu, self.fmean, tdense)   # T,N,q
		vt = tf.transpose(vt,[1,0,2]) # N,T,q
		st = tf.transpose(st,[1,0,2]) # N,T,q
		logpt = tf.transpose(logpt,[1,0]) # N,T
		vt_mu = tf.transpose(vt_mu,[1,0,2]) # N,T,q
		st_mu = tf.transpose(st_mu,[1,0,2]) # N,T,q
		def extractor(v_s_logp_vmu_smu_t):
			vt,st,logpt,vmu,smu,t = v_s_logp_vmu_smu_t
			idx = tf.where(tf.abs(tdense-tf.expand_dims(t,1))<1e-5)[:,1]
			vt = tf.gather(vt,[idx],0)[0]
			st = tf.gather(st,[idx],0)[0]
			logpt = tf.gather(logpt,[idx],0)[0]
			vmu = tf.gather(vmu,[idx],0)[0]
			smu = tf.gather(smu,[idx],0)[0]
			return vt, st, logpt, vmu, smu
		vt, st, logpt, vt_mu, st_mu = tf.map_fn(extractor,(vt,st,logpt,vt_mu,st_mu,self.t), \
								  dtype=(tf.float32,tf.float32,tf.float32,tf.float32,tf.float32)) # N,T,q
		self.vt = tf.reshape(tf.transpose(vt,[1,0,2]), [-1, self.q],name='vt_L') # T*N, q
		self.st = tf.reshape(tf.transpose(st,[1,0,2]), [-1, self.q],name='st_L') # T*N, q
		self.zt = tf.concat([self.vt,self.st],1) # T*N, 2q
		self.vt_mu = tf.reshape(tf.transpose(vt_mu,[1,0,2]),[-1, self.q],name='vt_mu') # T*N, q
		self.st_mu = tf.reshape(tf.transpose(st_mu,[1,0,2]),[-1, self.q],name='st_mu') # T*N, q
		self.logpt = tf.transpose(logpt,[1,0]) # T,N
		# reconstruct
		self.x_rec_mu,  self.x_rec_log_sigma_sq  = self.decode(self.st,dec_fnc) # T*N,D
		self.x_rec_mu_mu, self.x_rec_log_sigma_sq_mu = self.decode(self.st_mu,dec_fnc) # T*N,D
		self.x_rec_mu = tf.identity(self.x_rec_mu,name='x_rec_mu_L')
		self.x_rec_mu_mu = tf.identity(self.x_rec_mu_mu,name='x_rec_mu_mu')
		self.x_rec_log_sigma_sq = 0*tf.ones(tf.shape(self.x_rec_log_sigma_sq))
		if self.inst_enc_KL:
			self.inst_enc(x0vs_enc, x0ss_enc, v_enc_fnc, s_enc_fnc)


	def inst_enc(self, x0vs, x0ss, v_enc_fnc, s_enc_fnc): # (T-a_len)*N, enc_expected_input_shape
		self.vt_mu_enc, self.vt_log_sigma_sq_enc = v_enc_fnc(x0vs, self.q, self.v_bns, 'enc_v0') # (T-a_len)*N,q
		self.st_mu_enc, self.st_log_sigma_sq_enc = s_enc_fnc(x0ss, self.q, self.s_bns, 'enc_s0') # (T-a_len)*N,q
		zt_mu_enc = tf.concat([self.vt_mu_enc,self.st_mu_enc],1,name='zt_mu_enc') # (T-a_len)*N,2q
		zt_log_sigma_sq_enc = tf.concat([self.vt_log_sigma_sq_enc,self.st_log_sigma_sq_enc],1) # (T-a_len)*N,2q
		N = tf.shape(self.x)[0]
		zt_mu_enc = tf.reshape(zt_mu_enc,[-1,N,2*self.q]) # T-a_len,N,2q
		zt_log_sigma_sq_enc = tf.reshape(zt_log_sigma_sq_enc,[-1,N,2*self.q]) # T-a_len,N,2q
		self.zt_L_mu_enc = tf.reshape(tf.tile(zt_mu_enc,[1,1,self.L]), [-1, 2*self.q]) # (T-a_len)*N*L,2q
		self.zt_L_log_sigma_sq_enc = tf.reshape(tf.tile(zt_log_sigma_sq_enc,[1,1,self.L]), [-1, 2*self.q]) # (T-a_len)*N*L,2q
		T = tf.shape(zt_mu_enc)[0] # T-a_len
		N = tf.shape(zt_mu_enc)[1]
		self.zt_L_ode = self.zt[:T*N*self.L,:] # (T-a_len)*N*L,2q

	def enc_mnist_bball(self,x,nout,bn_list,scp='enc'):
		with tf.variable_scope(scp, reuse=tf.AUTO_REUSE):
			x = tf.nn.relu(bn_list[0](conv2d(x, self.NF_enc,   k_h=self.KW_enc, k_w=self.KW_enc, name='h0'),train=self.train))
			x = tf.nn.relu(bn_list[1](conv2d(x, self.NF_enc*2, k_h=self.KW_enc, k_w=self.KW_enc, name='h1'),train=self.train))
			x = tf.nn.relu(bn_list[2](conv2d(x, self.NF_enc*4, k_h=self.KW_enc, k_w=self.KW_enc, name='h2'),train=self.train))
			x = tf.nn.relu(bn_list[3](conv2d(x, self.NF_enc*8, k_h=self.KW_enc, k_w=self.KW_enc, name='h3'),train=self.train))
			x = tf.contrib.layers.flatten(x)
			z0_mu         = tf.contrib.layers.fully_connected(x, nout, activation_fn=None)
			z0_log_sig_sq = tf.contrib.layers.fully_connected(x, nout, activation_fn=None)
			return z0_mu, z0_log_sig_sq
	def enc_dense(self,x,nout,bn_list,scp='enc'):
		with tf.variable_scope(scp, reuse=tf.AUTO_REUSE):
			h1 = tf.contrib.layers.fully_connected(x,  100, activation_fn=tf.nn.relu)
			h2 = tf.contrib.layers.fully_connected(h1, 100, activation_fn=tf.nn.relu)
			z0_mu         = tf.contrib.layers.fully_connected(h2, nout, activation_fn=None)
			z0_log_sig_sq = tf.contrib.layers.fully_connected(h2, nout, activation_fn=None)
			return z0_mu, z0_log_sig_sq

	def dec_mnist_bball(self,x,scp='dec'):
		with tf.variable_scope(scp, reuse=tf.AUTO_REUSE):
			z = tf.reshape(x,[-1,self.q])
			if 'mnist' in self.task:
				z = tf.contrib.layers.fully_connected(z, 3*3*8, activation_fn=None)
				z = tf.reshape(z, shape=[-1, 3, 3, 8])
				z = tf.nn.relu(self.d_bns[0](deconv2d(z, self.NF_dec*4, k_h=3, k_w=3, padding='valid', name='h0'),train=self.train))
			elif self.task is 'bballs':
				z = tf.contrib.layers.fully_connected(z, 4*4*8, activation_fn=None)
				z = tf.reshape(z, shape=[-1, 4, 4, 8])
				z = tf.nn.relu(self.d_bns[0](deconv2d(z, self.NF_dec*4, k_h=self.KW_dec, k_w=self.KW_dec, name='h0'),train=self.train))
			z = tf.nn.relu(self.d_bns[1](deconv2d(z, self.NF_dec*2, k_h=self.KW_dec, k_w=self.KW_dec, name='h1'),train=self.train))
			z = tf.nn.relu(self.d_bns[2](deconv2d(z, self.NF_dec, k_h=self.KW_dec, k_w=self.KW_dec, name='h2'),train=self.train))
			z = deconv2d(z, 1, d_h=1, d_w=1, k_h=self.KW_dec, k_w=self.KW_dec, name='h3')
			logits = tf.nn.sigmoid(z)
			logits = tf.contrib.layers.flatten(logits)
		return logits, logits
	def dec_dense(self,x,scp='dec'):
		with tf.variable_scope(scp, reuse=tf.AUTO_REUSE):
			z = tf.reshape(x,[-1,self.q])
			z = tf.contrib.layers.fully_connected(z, 100, activation_fn=tf.nn.relu)
			z = tf.contrib.layers.fully_connected(z, 100, activation_fn=tf.nn.relu)
			x_mean       = tf.contrib.layers.fully_connected(z, self.D, activation_fn=None)
			x_log_sig_sq = tf.contrib.layers.fully_connected(z, self.D, activation_fn=None)
			return x_mean, x_log_sig_sq

	def ode_density(self, v0, s0, logP0, fmom, t, dt=None):
		"""runs the ode integrator from the mean specified by the encoder and also from random points
		   time_major=True, meaning the first index is time
				zt_mu   - latent trajectories where the initial values are the latent mean (encoder output)
				zt      - latent trajectories where the initial values are drawn from a Gaussian normal
						  whose mean and variances are encoder outputs
		"""
		def ode_density_step(v_s_logP,t): # N,2q+1
			v_s  = v_s_logP[:,:-1] # N,2q
			f1 = fmom(v_s,t) # N,q
			f2 = v_s[:,:self.q] # N,q
			df1dvs = ops.convert_to_tensor([tf.gradients(f1[:,d],v_s)[0] for d in range(self.q)]) # q,N,2q
			df1dv = df1dvs[:,:,:self.q] # q,N,q
			df1dv = tf.transpose(df1dv,[1,0,2]) # N,q,q
			dlogP = -tf.trace(df1dv)[:,None] # N,1
			dv_ds_dlogP = tf.concat([f1,f2,dlogP],1) # N,2q+1
			return dv_ds_dlogP
		dt = tf.constant(0.02,tf.float32)
		v0_s0_logp0 = tf.concat([v0,s0,logP0],1) # N,2q+1
		vt_st_logpt = tf.contrib.integrate.odeint_fixed(ode_density_step, v0_s0_logp0, t, dt) # T,N,2q+1
		vt    = tf.identity(vt_st_logpt[:,:,0:self.q], name="latent-velocity")
		st    = tf.identity(vt_st_logpt[:,:,self.q:2*self.q], name="latent")
		logpt = tf.identity(vt_st_logpt[:,:,-1], name="latent_density")
		return vt,st,logpt
	def ode(self, v0, s0, fmom, t, dt=None):
		"""runs the ode integrator from the mean specified by the encoder and also from random points
		   time_major=True, meaning the first index is time
				zt_mu   - latent trajectories where the initial values are the latent mean (encoder output)
				zt      - latent trajectories where the initial values are drawn from a Gaussian normal
						  whose mean and variances are encoder outputs
		"""
		dt = tf.constant(0.02,tf.float32)
		v0_s0 = tf.concat([v0,s0],1) # N,2q
		def ode_f_helper(v_s,t):
			f1 = fmom(v_s,t) # N,q
			f2 = v_s[:,:self.q] # N,q
			return tf.concat([f1,f2],1) # N,2q
		vt_st = tf.contrib.integrate.odeint_fixed(ode_f_helper, v0_s0, t, dt) # T,N,2q
		vt    = tf.identity(vt_st[:,:,0:self.q], name="latent-velocity-mean")
		st    = tf.identity(vt_st[:,:,self.q:2*self.q], name="latent-mean")
		return vt,st

	def decode(self, Z,dec_fnc):
		"""decodes all latent trajectories
		   input  - time_major = True
		   output - time_major = False
			x_rec_mean       - mean reconstruction of the trajectories whose initial values are sampled from
								 the encoder outputs [batch_size*L,T,D]
			x_rec_log_sig_sq - diagonal std of the reconstruction of the trajectories whose initial values are
								 sampled from the encoder outputs [batch_size*L,T,D]
		"""
		x_rec_mean, x_rec_log_sig_sq = dec_fnc(Z)
		x_rec_mean = tf.identity(x_rec_mean, name="rec_mean")
		x_rec_log_sig_sq = tf.identity(x_rec_log_sig_sq, name="rec_logvar")
		return x_rec_mean, x_rec_log_sig_sq

	def gauss_KL(self,mu1,mu2,logsigmasq,loglambdasq): # KL( N(mu1,diag(sig2)) || N(mu2,lambda2xI) )
		return 0.5 * tf.reduce_sum(-1 - logsigmasq  + loglambdasq + (tf.square(mu1-mu2)+tf.exp(logsigmasq))/tf.exp(loglambdasq), 1)
	def gauss_KL_I(self,mu,logsigmasq): # KL( N(mu,sig2) || N(0,I) )
		return self.gauss_KL(mu,0.0,logsigmasq,0.0)
	def rec_lhood(self, x, x_rec_mu, x_rec_log_sigma_sq):
		N = tf.shape(x)[0]
		x = tf.transpose(x,[1,0,2]) # T,N,D
		x = tf.reshape(tf.tile(x,[1,1,self.L]), [-1, self.D])
		if self.dec_out == 'bernoulli':
			lhood = tf.reduce_sum(x*tf.log(1e-5+x_rec_mu) + (1-x)*tf.log(1e-5+1-x_rec_mu),1)
		elif self.dec_out == 'normal':
			mvn = tfd.MultivariateNormalDiag(loc=x_rec_mu, scale_diag=tf.sqrt(1e-10 + tf.exp(x_rec_log_sigma_sq)))
			lhood = tf.reduce_sum(mvn.log_prob(x))
		return tf.reduce_sum(lhood) / tf.cast(N,tf.float32) / self.L # = 1/N * \sum_{i,t} x_t^i
	
	def create_loss(self):
		self.KL_w = 0
		if self.f_opt==2:
			for w in self.f_weight_means:
				self.KL_w += tf.reduce_sum(self.gauss_KL_I(w, self.loglambdasq))
			for w in self.f_bias_means:
				self.KL_w += tf.reduce_sum(self.gauss_KL_I(tf.reshape(w,[1,-1]), self.loglambdasq))
		self.mean_reconstr_lhood = self.rec_lhood(self.x, self.x_rec_mu_mu, self.x_rec_log_sigma_sq_mu)
		self.reconstr_lhood = self.rec_lhood(self.x, self.x_rec_mu, self.x_rec_log_sigma_sq)
		tf.summary.scalar('beta-KL', self.beta*self.KL_w)
		tf.summary.scalar('reconstr_lhood', self.reconstr_lhood)
		self.inst_enc_loss = tf.constant(0.0)
		N = tf.shape(self.x)[0]
		self.log_p = tfd.MultivariateNormalDiag(loc=tf.zeros(2*self.q), scale_diag=tf.ones(2*self.q)).log_prob(self.zt)
		self.log_p = tf.reduce_sum(self.log_p) / tf.cast(N,tf.float32) / self.L
		self.log_q = tf.reduce_sum(self.logpt) / tf.cast(N,tf.float32) / self.L
		tf.summary.scalar('log_p', self.log_p)
		tf.summary.scalar('log_q', self.log_q)
		self.elbo = -self.beta*self.KL_w + self.reconstr_lhood - self.log_q + self.log_p
		self.vae_loss = tf.multiply(-1.0,self.elbo,name='loss')
		if self.inst_enc_KL and self.gamma>0.0:
			mvn = tfd.MultivariateNormalDiag(loc=self.zt_L_mu_enc, scale_diag=tf.sqrt(tf.exp(self.zt_L_log_sigma_sq_enc)))
			self.qenc_zt_ode = tf.reduce_sum(mvn.log_prob(self.zt_L_ode)) / tf.cast(N,tf.float32) / self.L
			self.inst_enc_loss = self.gamma * (self.log_q - self.qenc_zt_ode)
			self.vae_loss += self.inst_enc_loss
			print('inst_enc_KL loss added to VAE loss')
		tf.summary.scalar('inst_enc_loss', self.inst_enc_loss)

	def create_optimizers(self):
		self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
		self.expdec = tf.train.exponential_decay(self.eta,self.global_step,200,0.995,staircase=True)
		vae_vars = tf.trainable_variables()
		numpars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
		print('{:d} total trainable parameters'.format(numpars))
		for i in vae_vars:
			print(i)
		self.vae_optimizer = tf.train.AdamOptimizer(self.expdec).minimize(self.vae_loss, \
			name='adam_vae',var_list=vae_vars,global_step=self.global_step)
		self.elbo_optimizer = tf.train.AdamOptimizer(self.expdec).minimize(-self.elbo, \
			name='adam_vae_elbo',var_list=vae_vars,global_step=self.global_step)
		self.vae_rec_optimizer = tf.train.AdamOptimizer(self.expdec).minimize(-self.reconstr_lhood, \
			name='adam_vae_rec',var_list=vae_vars,global_step=self.global_step)
		
	def integrate(self,X,dt=0.1):
		t = dt * np.arange(0,X.shape[1],dtype=np.float32)
		t = np.tile(t,[X.shape[0],1])
		st = self.sess.run((self.st), feed_dict={tf.get_default_graph().get_tensor_by_name('X:0'):X,\
														  tf.get_default_graph().get_tensor_by_name('t:0'):t,
														  tf.get_default_graph().get_tensor_by_name('Tss:0'):X.shape[1],\
														  self.train:False})
		st = np.reshape(st,(-1,X.shape[0],self.q)) # [T,N,D]
		st = np.transpose(st,[1,2,0]) # [N,D,T]
		return st
	def reconstruct(self,X,dt=0.1):
		t = dt * np.arange(0,X.shape[1],dtype=np.float32)
		t = np.tile(t,[X.shape[0],1])
		Xrec = self.sess.run(self.x_rec_mu_mu, feed_dict={tf.get_default_graph().get_tensor_by_name('X:0'):X,\
														  tf.get_default_graph().get_tensor_by_name('t:0'):t,
														  tf.get_default_graph().get_tensor_by_name('Tss:0'):X.shape[1],\
														  self.train:False})
		Xrec = np.reshape(Xrec,(-1,X.shape[0],self.D)) # [T,N,D]
		Xrec = np.transpose(Xrec,[1,0,2]) # [N,T,D]
		return Xrec
	def save_model(self,ckpt_dir,ext='noname'):
		saver = tf.train.Saver(save_relative_paths=True)
		saver.save(self.sess, os.path.join(ckpt_dir, self.task, ext))












