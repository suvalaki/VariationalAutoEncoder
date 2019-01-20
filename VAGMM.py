import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def log_normal_pdf(x,mean,var, minvar, axis=-1):
	if var < minvar:
		var = tf.max(var, minvar, name='max_var')	
	return -0.5 * tf.reduce_sum(
        tf.log(2 * np.pi) + tf.log(var) + tf.square(x - mean) / var, axis)


def graph_Qygx(
	x, 
	encode_layer_dims = [500,500], 
	encode_layer_acti = [tf.nn.tanh, tf.nn.tanh],
	categorical_dims  = 10,
	):

	with tf.variable_scope('Qygx'):
		with tf.variable_scope('Encoding'):
			for i, (dims_, acts_) in enumerate(
				zip(encode_layer_dims, encode_layer_acti)):
				x = tf.layers.dense(x,units=dims_,activation=acts_,
					name=f'Layer_{i}', reuse=True)
		with tf.variable_scope('Latent'):
			qygz_logits = tf.layers.dense(
				x,units=categorical_dims, activation=None, 
				name='logits', reuse=True)
			qygz = tf.nn.softmax(qygz_logits, name='prob')
			

	return qygz_logits, qygz

def graph_Qz_kgxy_k(
	x,y_k,
	encode_layer_dims = [500,500], 
	encode_layer_acti = [tf.nn.tanh, tf.nn.tanh],
	latent_dims		  = 64
	):

	xy_k = tf.concat([x,y_k], axis = -1, name='xy_concat')

	with tf.variable_scope('Qzgxy'):
		with tf.variable_scope('Encoding'):
			for i, (dims_, acts_) in enumerate(
				zip(encode_layer_dims, encode_layer_acti)):
				xy_k = tf.layers.dense(xy_k,units=dims_,activation=acts_,
					name=f'Layer_{i}', reuse=True)
		with tf.variable_scope('Latent'):
			mu_k = tf.layers.dense(inputs=xy_k, 
				units=latent_dims, activation=None, name='mean')
			var_k = tf.layers.dense(inputs=xy_k, units=tf.nn.softplus, 
				activation=None, name='var')
			sd_k = tf.sqrt(var_k, name='sd')

			dist_z_k = tfp.distributions.Normal(loc=mu_k, scale=sd_k, 
				name='distribution')
			z_k = dist_z_k.sample()
			qzgxy_k = dist_z_k.prob(z_k)
			log_qzgxy_k = dist_z_k.log_prob(z_k)

	return z_k, qzgxy_k, log_qzgxy_k


def graph_Qzygx(
	x, 
	y_k_encode_layer_dims = [500,500],
	y_k_encode_layer_acti = [tf.nn.tanh, tf.nn.tanh],
	y_k_categorical_dims  = 10,
	z_k_encode_layer_dims = [500,500], 
	z_k_encode_layer_acti = [tf.nn.tanh, tf.nn.tanh],
	z_k_latent_dims = 64
	):

	k = y_k_categorical_dims

	with tf.variable_scope('Qzygx'):

		# binarize data. create a y "placeholder". Get the categorical vectors y
		with tf.name_scope('x_binarized'):
			xb = tf.cast(tf.greater(x, tf.random_uniform(tf.shape(x), 0, 1)), tf.float32)
		with tf.name_scope('y_placeholder'):
			y_ = tf.fill(tf.pack([tf.shape(x)[0], k]), 0.0)
		with tf.variable_scope('y_one_hot'):
			y  = [None] * k
			for i in range(k):
				y[i] = tf.add(y_, tf.Constant(np.eye(k)[i], name=f'hot_at_{i}'))

		# propose distribution over y
		qy_logit, qy = graph_Qygx(
			xb, y_k_encode_layer_dims, y_k_encode_layer_acti,k)

		# for each proposed y, infer z and reconstruct x
		# For each categorical y create the subgraph got z_k_gxy_k
		with tf.variable_scope('Qzgxy'):
			#z_k is z dimensioned by y_k
			z_k, qzgxy_k, log_qzgxy_k = [[None] * k for i in range(3)]
			for i in range(k):
				z_k[i], qzgxy_k[i], log_qzgxy_k[i] = graph_Qz_kgxy_k(x,y_k, 
					z_k_encode_layer_dims, z_k_encode_layer_acti, z_k_latent_dims)
					

	return xb, y, qy_logit, qy, z_k, qzgxy_k, log_qzgxy_k
		

def graph_Pz_kgy_k(y_k, z, latent_dims=64):

	with tf.variable_scope('Pz_kgy_k'):
		prior_z_mu_k = tf.layer.dense(y_k, units=latent_dims, 
			activation=None, name='mu')
		prior_z_var_k = tf.layer.dense(y_k, units=latent_dims,
			activation=tf.nn.softplus, name='var')
		prior_z_logvar = tf.log(prior_z_var_k, name='logvar')
		prior_z_sd = tf.sqrt(prior_z_var_k, name='sd')

		prior_z_k_distribution = tfp.distributions.Normal(
			loc=prior_z_mu_k, scale=prior_z_sd, name='distribution')

		prior_pz_kgy_k 	= prior_z_k_distribution.prob(z)
		log_pz_kgy_k 	= prior_z_k_distribution.log_prob(z)

	return prior_z_mu_k, prior_z_var_k, prior_pz_kgy_k, log_pz_kgy_k
		

def graph_Px_g_z(z, encode_layer_dims, encode_layer_acti, output_dim):

	with tf.variable_scope('Px_g_z('):
		with tf.variable_scope('Decoder'):
			for i, (dims_, acts_) in enumerate(
				zip(encode_layer_dims, encode_layer_acti)):
				z = tf.layers.dense(z,units=dims_,activation=acts_,
					name=f'Layer_{i}', reuse=True)
		with tf.variable_scope('px'):
			px_logit = tf.layers.dense(z, units=output_dim, activation=None, 
				name='logit',reuse=True)
			px = tf.nn.softmax(px_logit, name='prob')

	return px_logit, px


def loss(py, pzgy, pxgyz, qygx, qzgxy):	

	with tf.variable_scope('loss'):

		with tf.variable_scope('term1'):
			t1 = tf.log(py) - tf.reduce_sum(qygx * tf.log(qygx), axis=-1)

		with tf.variable_scope('term2'):
			t2 = tf.reduce_sum( qygx * tf.reduce_mean(
				qzgxy * ( tf.log(pzgy) - tf.log(qzgxy) ) , axis=-1
			) , axis=-1)
		with tf.variable_scope('term2'):
			t3 = tf.reduce_sum( qygx * tf.reduce_mean(
				qzgxy * ( tf.log(qygx) ) , axis=-1
			) , axis=-1)

	with tf.variable_scope('total_loss'):
		total_loss = t1 + t2 + t3
	
	return total_loss


def graph_build(	
	y_k_encode_layer_dims = [500,500],
	y_k_encode_layer_acti = [tf.nn.tanh, tf.nn.tanh],
	y_k_categorical_dims  = 10,
	z_k_encode_layer_dims = [500,500], 
	z_k_encode_layer_acti = [tf.nn.tanh, tf.nn.tanh],
	z_k_latent_dims = 64,
	output_dim = 784
	):

	with tf.Graph().as_default() as graph:
		x = tf.placeholder(tf.float32, (None, output_dim))

		# Encoder
		xb, y, qy_logit, qy, z_k, qzgxy_k, log_qzgxy_k = graph_Qzygx(
			x, y_k_encode_layer_dims, y_k_encode_layer_acti, y_k_categorical_dims, 
			z_k_encode_layer_dims, z_k_encode_layer_acti,z_k_latent_dims)

		with tf.variable_scope('Pzgy'):
			(prior_z_mu_k, prior_z_var_k, 
			prior_pz_kgy_k, log_pz_kgy_k) = [[None] * y_k_categorical_dims 
											for i in range(4)]
			for i in range(y_k_categorical_dims):
				(prior_z_mu_k[i], prior_z_var_k[i], 
				prior_pz_kgy_k[i], log_pz_kgy_k[i]) = graph_Pz_kgy_k(
					y[i], z_k[i], z_k_latent_dims)

		# Priors
		prior_y_pi = tf.constant(1/y_k_categorical_dims)

		px_logit, px = graph_Px_g_z(
			z_k, z_k_encode_layer_dims, z_k_encode_layer_acti, output_dim)

		# Binerised Output
		output = tf.cast(
			tf.greater(px, tf.random_uniform(tf.shape(px), 0, 1)), tf.float32)

		# Loss
		vaeloss = loss(py=prior_y_pi, pzgy=prior_pz_kgy_k, 
			pxgyz=px, qygx=qy, qzgxy=qzgxy_k)

		# Minimizer
		minimizer = tf.train.AdamOptimizer().minimize(vaeloss)


		# Create tensorflow session and initilize
		init = tf.global_variables_initializer()
		sess = tf.Session(graph=graph)
		sess.run(init)

		return sess





