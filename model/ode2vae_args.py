import argparse
import os
import tensorflow as tf

class ODE2VAE_Args:
    '''
    Arguments for data, model, and checkpoints.
    '''
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Inputs to the DVAE model')

        self.parser.add_argument('--data_root', type=str, default='data',
                        help='root of the data folder')
        self.parser.add_argument('--ckpt_dir', type=str, default='checkpoints',
                        help='checkpoints folder')
        self.parser.add_argument('--task', type=str, default='mocap_many',
                        help='experiment to execute')
        self.parser.add_argument('--f_opt', type=int, default=2,\
                        help='neural network(1), Bayesian neural network(2)')
        self.parser.add_argument('--amort_len', type=int, default=3,\
                        help='the number data points (from time zero) velocity encoder takes as input')
        self.parser.add_argument('--activation_fn', type=str, default="relu",\
                        help='activation function used in fully connected layers ("relu","tanh","identity")')
        self.parser.add_argument('--q', type=int, default=10,
                        help='latent dimensionality')
        self.parser.add_argument('--gamma', type=float, default=1.0,\
                        help='constant in front of variational loss penalty (sec. 3.3)')
        self.parser.add_argument('--inst_enc_KL', type=int, default=1,\
                        help='(1) if use variational loss penalty (sec. 3.3); (0) otherwise')
        self.parser.add_argument('--Hf', type=int, default=100,
                        help='number of hidden units in each layer of differential NN')
        self.parser.add_argument('--He', type=int, default=50,
                        help='number of hidden units in each layer of encoder')
        self.parser.add_argument('--Hd', type=int, default=50,
                        help='number of hidden units in each layer of decoder')
        self.parser.add_argument('--Nf', type=int, default=2,
                        help='number of hidden layers in differential NN')
        self.parser.add_argument('--Ne', type=int, default=2,
                        help='number of hidden layers in encoder')
        self.parser.add_argument('--Nd', type=int, default=2,
                        help='number of hidden layers in  decoder')
        self.parser.add_argument('--NF_enc', type=int, default=16,\
                        help='number of filters in the first encoder layer')
        self.parser.add_argument('--NF_dec', type=int, default=32,\
                        help='number of filters in the last decoder layer')
        self.parser.add_argument('--KW_enc', type=int, default=5,\
                        help='encoder kernel width')
        self.parser.add_argument('--KW_dec', type=int, default=5,\
                        help='decoder kernel width') 
        self.parser.add_argument('--eta', type=float, default=0.001,\
                        help='learning rate')
        self.parser.add_argument('--batch_size', type=int, default=10,\
                        help='number of sequences in each training mini-batch')
        self.parser.add_argument('--num_epoch', type=int, default=1000,\
                        help='number of training epochs')
        self.parser.add_argument('--subject_id', type=int, default=0,\
                        help='subject ID in mocap_single experiments')

    def parse(self):
        opt = self.parser.parse_args()
        opt = vars(opt)
        if opt["activation_fn"]=='relu':
            opt["activation_fn"] = tf.nn.relu
        elif opt["activation_fn"]=='tanh':
            opt["activation_fn"] = tf.nn.tanh
        elif opt["activation_fn"]=='identity':
            opt["activation_fn"] = tf.identity
        else:
            raise ValueError('activation_fn must be either of "relu","tanh","identity": {:s}'.format(opt["activation_fn"]))
        return opt
