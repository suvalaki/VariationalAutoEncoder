# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 11:51:10 2019

@author: david
"""

import tensorflow as tf
import tensorflow_probability as tfp

MultivariateNormalDiag = tfp.distributions.MultivariateNormalDiag

def plog(x):
    return tf.maximum(x, 10e-6)


def multivariate_normal_pdf(distribution, tensor):
    
    prob = tf.map_fn(lambda x: tf.map_fn( lambda y: distribution.log_prob(y), x ), tensor )
    
    return prob
    


class GMVAE():
    
    # https://arxiv.org/pdf/1611.05148.pdf
    
    def __init__(self):
        
        #number of batches to pass to the optimiser
        self.B = 200
        # Number of Monte Carlo Samples
        self.L = 1
        # Number of dimensions of X
        self.D = 784
        # Number of clusters
        self.K = 3
        # Encoding Layers dimensions
        self.encoding_dims = [500, 100]
        # activation functions for encoding layers
        self.encoding_act = [tf.nn.relu, tf.nn.relu] 
        # Latent Dimension. dimension of mu_c, var_c, mu_tilde and var_tilde
        self.J = 2


        #
        self.scale_prior = 10

        # division max opperator
        self.small_value = 10e-6
        self.div_prot = lambda x: tf.maximum(x, self.small_value)
        
        
        # Prevent training until after entropy
        self.lambda_1 = -100
        self.lambda_2 = 20
        
        pass
    
    def _graph_q_cgx(self):
        """
        From appendix (A) we have q(c|x) = E_{q(z|x)}[p(c|z)]
        
        Input
        ================
        self.p_c : tf.tensor ; shape (n_clusters,); 
                    Prior distribution of C~cat(pi)
        self.p_zgc : tf.tensor; shape (n_samples, batch, n_clusters)
                Prior probability for z|c as given by Normal density, z_mu_prior,
                and z_var_prior for each c
        
        """
        
        
        
        self.numerator_c = self.p_c_pz
        self.denominator = tf.reduce_sum(self.p_c_pz, axis = 0)
        return (  tf.reduce_mean( self.numerator_c / 
                        self.div_prot(self.denominator[None,:,:]),
                                         axis = 1 )   )
        
    
    def _graph_g(self, x_in):
        """
        Creates the neural network g(.) which replicates q(z|x)

        Output
        ====================
        mu_tilda :
        log_var_tilda :
        
        
        Explain
        ====================
        Produces [mu_tilda ; log_var_tilda ] = g(x; phi) : g being the 
            neural network with parametes phi
        Q(z|x) = N( z; mu_tilda, var_tilda * I )
        
        """
        
        x = x_in
        with tf.variable_scope("Encoder"):
                        
            z = x
            n_layer = 0
            for size, activation in zip(self.encoding_dims,self.encoding_act):
                 n_layer += 1
                 z = tf.layers.dense(z, size, activation=activation,
                    name=f"layer_{n_layer}")
            
            self.temp = z
            
            with tf.variable_scope("Latent"):
                # activation function of latent layer is linear
                
                mu = tf.layers.dense(z, self.J, activation=None,
                        name=f"mu")
                log_var = tf.layers.dense(z, self.J, activation=None,
                        name=f"log_var")
                var = tf.exp(log_var, name = 'var')
                scale = tf.sqrt(var, name = "scale")
                
                #z_distribution = MultivariateNormalDiag(mu, scale)
                #z = z_distribution.sample(self.L)
                
                # Since the Monte Carlo estimate of the expectation above is 
                # non-differentiable w.r.t φ when z(l) is directly sampled 
                # from z ∼ N (µ˜,σ˜2I), we use the reparameterization trick to 
                # obtain a differentiable estimation:
                epsilon = MultivariateNormalDiag([0] * self.J, scale_diag = [1] * self.J)
                epsilon_sample = epsilon.sample(self.L)
                z = mu[None,:,:] + scale[None,:,:] * epsilon_sample[:,None,:]
                
                
                
                
                q_zgx = tf.map_fn(epsilon.prob, epsilon_sample)
                #q_zgx = multivariate_normal_pdf(z_distribution, z)
                
                p_z = [tf.map_fn(
                        self.z_prior_distribution[k].prob, z) 
                        for k in range(self.K)]
                p_c_pz = tf.convert_to_tensor(
                        [ self.p_c[k] * p_z[k] for k in range(self.K)])
                
                self.pc_temp =  self.p_c
                
                p_z = tf.convert_to_tensor(p_z)
                
                #p_z = z.
                
        return mu, log_var, var, scale, z, q_zgx, p_z, p_c_pz
            
    def _graph_f(self, z_in):
        """
        Computes the expectation vector for the MixtureGaussian
        """
        
        
        z = z_in
        with tf.variable_scope("Decoder"):
                        
            x=z
            n_layer = 0
            for size, activation in zip(list(reversed(self.encoding_dims)),
                                        list(reversed(self.encoding_act))):
                 n_layer += 1
                 x = tf.layers.dense(x, size, activation=activation,
                    name=f"layer_{n_layer}")
                 
            mu_x = tf.layers.dense(z, self.D, name = 'mu_x')
            x_out = tf.nn.softmax(mu_x, name='x_decode')
            
        return mu_x, x_out
    
    
    
    def _build(self):
        
        
        tf.reset_default_graph()
        
        with tf.variable_scope("priors"):
        
            # Prior probability of cluster c. P(c|pi)
            self.p_c = tf.Variable(tf.convert_to_tensor([1/self.K] * self.K ), trainable=False)
            # Prior mean for latent centroids. Initialised at zero for all clusters
            self.z_mu_prior = tf.Variable(tf.zeros(shape=(self.K,self.J)), trainable=False)
            # Prior variance for latent centroids. Initialised at 1
            self.z_var_prior = tf.Variable(self.scale_prior * tf.ones(shape=(self.K,self.J)), trainable=False)
            self.z_scale_prior = tf.Variable(tf.sqrt(self.z_var_prior), trainable=False)
            # note scale = sd
            self.z_prior_distribution = [
                    MultivariateNormalDiag(
                            loc=self.z_mu_prior[k,:], 
                            scale_diag=tf.sqrt(self.z_scale_prior[k,:]),  
                    ) for k in range(self.K)]
                
        self.inputs = tf.cast(
                tf.placeholder(dtype='float',shape=(None,self.D ), name='x'),
                'float32')
        
        #encoder network
        (self.mu_tilde, self.log_var_tilde, self.var_tilde, self.scale_tilde, 
         self.z_tilde, self.q_zgx, self.p_z, 
         self.p_c_pz) = self._graph_g(self.inputs)
        
        #decoder network
        self.mu_x, self.output = self._graph_f(self.z_tilde)
        # outputs (samples, batch, output_dim)
        
        
        
        #log_p_zgc = [self.z_prior_distribution.log_prob[k](z_tilde)
        # probabilities
        self.q_cgx = self._graph_q_cgx()
        self.q_cgx_t = tf.transpose(self.q_cgx)
        self.cgx = tf.nn.softmax(tf.transpose(self.q_cgx))
        
        with tf.variable_scope('loss'):
        
            self.loss_xent = self.xent()
            self.loss_z = self.z_loss()
            self.loss_cat = self.cat_loss()
            
            self.loss_xent_sum = tf.reduce_mean(self.xent())
            self.loss_z_sum = tf.reduce_mean(self.z_loss())
            self.loss_cat_sum = tf.reduce_mean(self.cat_loss())
            
        self.cost  = self.loss_xent_sum + self.loss_z_sum + self.loss_cat_sum
            
        self.minimizer = tf.train.AdamOptimizer().minimize(self.cost )
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
    
    def xent(self):
        
        with tf.variable_scope('xent'):
            xent = tf.reduce_mean(
                tf.reduce_sum(
                    (self.inputs[None,:,:] * plog( self.small_value+ self.mu_x[:,:,:] ) +
                    (1-self.inputs[None,:,:]) * plog(self.small_value+ 1 - self.mu_x[:,:,:])),
                 axis = 2
                 ), 
                axis = 0 , name = 'xent'
            )
        
        return xent
    
    def z_loss(self):
        
        with tf.variable_scope('z_loss'):
            
            mu_p = self.z_mu_prior[None, :, :]
            v_p = self.z_var_prior[None, :, :]
            
            mu_t = self.mu_tilde[:,None,:]
            v_t = self.var_tilde[:,None,:]
            
            q_cgx = tf.transpose(self.q_cgx, (1,0))
            
            integrad = (tf.log(v_p) + v_t / self.div_prot(v_p) 
                            + tf.square(mu_t - mu_p) / self.div_prot(v_p))
            
            # outer reduction is for batches
            
            return -0.5 * tf.reduce_sum( q_cgx * tf.reduce_sum(
                integrad, axis = 2), axis = 1 )
            
    def cat_loss(self):
                
        return (tf.reduce_sum(self.q_cgx * plog( self.p_c[:,None] / self.div_prot(self.q_cgx) ), 0)
                + 0.5 * tf.reduce_sum( (1+plog(self.var_tilde)) , 1) )
        
    def train(self, mnist, epoch_size, n_batch):
        for epoch in range(epoch_size):
        
            self.sess.run(self.minimizer, feed_dict={'x:0': mnist.train.next_batch(n_batch)[0]})
        
            if (epoch + 1) % 100 == 0:
                loss_val, loss_xent_val, loss_z_val, loss_cat_val = (
                    self.sess.run(
                        [self.cost, self.loss_xent_sum, self.loss_z_sum,self.loss_cat_sum], 
                        feed_dict={'x:0': mnist.train.next_batch(n_batch)[0]}))
                print(f" epoch {epoch+1}/{epoch_size} : loss = {loss_val:.3f} " 
                    f"xent = {loss_xent_val:.3f} zloss = {loss_z_val:.3f} closs = {loss_cat_val:.3f}")


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
model = GMVAE()
model._build()

"""
model.z_tilde
model.mu_tilde

model.mu_x
model.z_prior_distribution
model.p_c_pz

model._graph_q_cgx()
model.loss_xent

model.loss_cat

model.sess.run(model.minimizer, feed_dict={'x:0': mnist.train.next_batch(10000)[0]})

mu_tilde
var_tilde
z_tilde
q_zgx

q_cgx
q_cgx_t
p_z

cost
loss_xent
loss_cat_sum
self.loss_z_sum

model.sess.run([model.cost,model.loss_xent_sum, model.loss_z_sum, model.loss_cat_sum], 
               feed_dict={'x:0': mnist.train.next_batch(10)[0]})

model.sess.run(model.q_cgx, 
               feed_dict={'x:0': mnist.train.next_batch(10)[0]})

model.sess.run(model.var_tilde, 
               feed_dict={'x:0': mnist.train.next_batch(10)[0]})

model.sess.run(model.p_z, 
               feed_dict={'x:0': mnist.train.next_batch(10)[0]})

[n.name for n in tf.get_default_graph().as_graph_def().node][1:600]

model.sess.run(model.loss_z, 
               feed_dict={'x:0': mnist.train.next_batch(1000)[0]})

"""

model.train( mnist, epoch_size=10, n_batch=1000)

       