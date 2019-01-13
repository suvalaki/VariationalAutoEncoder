# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 21:32:47 2019

@author: david
"""
import numpy as np
import tensorflow as tf

# https://github.com/tnakae/DAGMM/blob/master/dagmm/gmm.py
class DAGMM():
    
    # deep autoencoding gaussian mixture model for unsupervised anomoly detection
    # bo zong, qi song, martin renquang min
    #2018
    
    def __init__(self):
        self.lambda1 = 0.1
        self.lambda2 = 0.005
    
    @staticmethod
    def estimation_network(zc, zr, categorical_dim, latent_dim):
        
        """
        Note n_samples is interchangable with batch size
        
        Parameters
        ----------
        zc: tf.Tensor shape : ( n_samples , n_features )
            The reduced low-dimensional representation learned by the deep 
            autoencoder.
        zr: tf.Tensor shape : (  ,  )
            zr = f(x,x') where f denotes the function of calculating reconstruction error features. 
                It is a vector of distance metrics. Euclidiean distance, other distances ect. 
                We must define f ourselves
        """
        
        with tf.variable_scope('estimation_network'):
            
            z = tf.concat([zc,zr], axis = 1, name='z')
            
            h1 = tf.layers.dense(z, latent_dim, activation=tf.nn.relu, name='layer1')
            h2 = tf.layers.dense(h1, latent_dim, activation=tf.nn.relu, name='layer2')
            
            pi_logit = tf.layers.dense(h2, categorical_dim, activation=None, name='logit')
            gamma = tf.nn.softmax(pi_logit, name = 'prob') # soft mixture component membership prediction
            
            
        with tf.variable_scope('gmm_parameters'):
            # i   : index of samples
            # k   : index of components
            # l,m : index of features
            
            gamma_sum = tf.reduce_sum(gamma, axis = 0)
            phi = tf.reduce_mean(gamma, axis = 0, name='phi') #mean of batches
            mu = tf.einsum('ik,il->kl',  gamma, z) / gamma_sum[:,None]
            z_centred = (z[:,None,:] - mu[None,:,:])
            sigma = tf.einsum(
                'ikl,ikm->klm', z_centred, z_centred) / gamma_sum[:,None,None]

        return z, h1, h2, pi_logit, gamma, phi, mu, sigma
            
            
    
    @staticmethod
    def cholsky_decomp(z, mu, sigma):
        
        with tf.variable_scope("GMM_Cholskey"):
             # use Cholesky decomposition instead  of invers
            # for symtric and positive definite it solve Ax=b => x=a^{-1}b
            # https://math.stackexchange.com/questions/2422012/solving-a-linear-system-with-cholesky-factorization
            L = tf.Variable(tf.zeros(
                    shape=[mu.shape[0], z.shape[1], z.shape[1]]),
                    dtype=tf.float32, name="L")
            z_centered = z[:,None,:] - mu[None,:,:]
            #v = tf.matrix_triangular_solve(L, tf.transpose(z_centered, [1, 2, 0]))  # kli
            min_vals = tf.diag(tf.ones(z.shape[1], dtype=tf.float32)) * 1e-6
            L = tf.cholesky(tf.cast(sigma,'float32') + min_vals[None,:,:])
            
        return L
    

    def sample_energy(self, phi, z, mu, sigma):
        with tf.variable_scope("GMM_energy"):
            L = self.cholsky_decomp(z, mu, sigma)
            
            
             # Instead of inverse covariance matrix, exploit cholesky decomposition
            # for stability of calculation.
            z_centered = z[:,None,:] - mu[None,:,:]  #ikl
            v = tf.cholesky_solve(tf.cast(L,'float32'), tf.cast(tf.transpose(z_centered, [1, 2, 0]),'float32'))  # kli

            # log(det(Sigma)) = 2 * sum[log(diag(L))]
            log_det_sigma = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)), axis=1)

            # To calculate energies, use "log-sum-exp" (different from orginal paper)
            d = z.get_shape().as_list()[1]
            logits = tf.log(phi[:,None]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                + d * tf.log(2.0 * np.pi) + log_det_sigma[:,None])
            energies = - tf.reduce_logsumexp(logits, axis=0)
            
        return energies
    

    def cov_diag_loss(self, sigma):
        with tf.variable_scope("GMM_diag_loss"):
            diag_loss = tf.reduce_sum(tf.divide(1, tf.matrix_diag_part(sigma)))
        return diag_loss
    
    
    
    def reconstruction_error(self, x, x_dash):
        with tf.variable_scope('error'):
            return tf.reduce_mean(tf.reduce_sum(
                tf.square(x - x_dash), axis=1), axis=0)
                

    @staticmethod
    def compression_network(inputs, encoding_dim, categorical_dim, latent_dim):
        
        inputs_dim = int(inputs.shape[1])
        
        with tf.variable_scope("compression_network"):
            
            with tf.variable_scope("encoder"):
                h1 = tf.layers.dense(inputs, encoding_dim, activation=tf.nn.relu, name='layer1')
                h2 = tf.layers.dense(h1, encoding_dim, activation=tf.nn.relu, name='layer2')
                
            with tf.variable_scope("latent"):
                zc = tf.layers.dense(h2, latent_dim, activation=None, name='layer1')
        
            with tf.variable_scope("decoder"):
                d1 = tf.layers.dense(zc, encoding_dim, activation=tf.nn.relu, name='layer1')
                d2 = tf.layers.dense(d1, encoding_dim, activation=tf.nn.relu, name='layer2')
                
            with tf.variable_scope("output"):
                d3 = tf.layers.dense(d2, inputs_dim, activation=None, name='logit')
                output = tf.nn.softmax(d3, name='output')

        return h1,h2,zc,d1,d2,d3,output
            
    
    @staticmethod
    def distance_metrics(labels, predictions):
        
        def euclid_norm(labels):
            return tf.sqrt(tf.reduce_sum(tf.square(labels), axis=1))

        # Calculate Euclid norm, distance
        norm_x = euclid_norm(labels)
        norm_x_dash = euclid_norm(labels)
        dist_x = euclid_norm(labels - predictions)
        dot_x = tf.reduce_sum(labels * predictions, axis=1)

        # Based on the original paper, features of reconstraction error
        # are composed of these loss functions:
        #  1. loss_E : relative Euclidean distance
        #  2. loss_C : cosine similarity
        min_val = 1e-3
        loss_E = dist_x  / (norm_x + min_val)
        loss_C = 0.5 * (1.0 - dot_x / (norm_x * norm_x_dash + min_val))
        return tf.concat([loss_E[:,None], loss_C[:,None]], axis=1)
            
    
    def build_model(self, input_dim, encoding_dim, 
                    categorical_dim, latent_dim):
        
        
        tf.reset_default_graph()
        
        inputs = tf.cast(
                tf.placeholder(dtype='float',shape=(None,input_dim), name='x'),
                'float32') #cholsky requires float 32
        
        #Compression Network
        with tf.variable_scope("compression_network"):
            h1,h2,zc,d1,d2,d3,output = self.compression_network(
                    inputs, encoding_dim, categorical_dim, latent_dim)
    
        self.latent = zc
    
        with tf.variable_scope('reconstruction_error'):
            zr = self.distance_metrics(inputs, output)
            
        # estimation network
        
        with tf.variable_scope('estimation_network'):
            z, h1, h2, pi_logit, gamma, phi, mu, sigma = self.estimation_network(
                    zc, zr, categorical_dim, latent_dim)
            
            energy = self.sample_energy(phi, z, mu, sigma)
        
        # Loss function
        with tf.variable_scope('loss'):
            self.loss = (self.reconstruction_error(inputs, output) +
                    self.lambda1 * tf.reduce_mean(energy, axis = 0) +
                    self.lambda2 * self.cov_diag_loss(sigma))
            
        
        self.minimizer = tf.train.AdamOptimizer().minimize(self.loss)
        # Create tensorflow session and initilize
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
        
    def train(self, mnist, epoch_size, n_batch):
        for epoch in range(epoch_size):
        
            self.sess.run(self.minimizer, feed_dict={'x:0': mnist.train.next_batch(n_batch)[0]})
        
            if (epoch + 1) % 100 == 0:
                loss_val = self.sess.run(self.loss, 
                            feed_dict={'x:0': mnist.train.next_batch(n_batch)[0]})
                print(f" epoch {epoch+1}/{epoch_size} : loss = {loss_val:.3f}")