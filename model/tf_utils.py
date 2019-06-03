import tensorflow as tf

class batch_norm:
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon  = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,
						  decay=self.momentum,
						  updates_collections=None,
						  epsilon=self.epsilon,
						  scale=True,
						  is_training=train,
						  scope=self.name)

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='same', name="enc"):
	with tf.variable_scope(name):
		return tf.layers.conv2d(input_, output_dim, [k_h,k_w], [d_h,d_w], padding=padding)

def deconv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='same', name="dec"):
	with tf.variable_scope(name):
		return tf.layers.conv2d_transpose(input_, output_dim, [k_h,k_w], [d_h,d_w], padding=padding)