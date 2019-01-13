# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 22:20:44 2019

@author: david
"""

import tensorflow as tf
tfd = tf.contrib.distributions

sess = tf.Session()

with sess.as_default():
    # Initialize a single 2-variate Gaussian.
    K=3
    J=2
    mvnlist = []
    mvn = tfd.MultivariateNormalDiag(
        loc=[0.0] * J,
        scale_diag=[1.0] * J)
    mvnlist.append(mvn)
    
    z = tf.ones(shape = (K,2,J), dtype=tf.float32)
    p_z = tf.map_fn(mvnlist[0].prob, z)
    
    print(p_z.eval())
    z_mu_prior = tf.convert_to_tensor([ [0.0] * J ] * K)
    # Prior variance for latent centroids. Initialised at 1
    z_var_prior = tf.convert_to_tensor([ [1.0] * J ] * K)
    z_scale_prior = tf.sqrt(z_var_prior)
    
    z= tf.convert_to_tensor([[13.4104595, 12.128497]])
    
    z_prior_distribution = [
        tfd.MultivariateNormalDiag(
                loc=z_mu_prior[k,:], 
                scale_diag=tf.sqrt(z_scale_prior[k,:]),  
        ) for k in range(K)]
    
    p_z = [tf.map_fn(
            z_prior_distribution[k].prob, z) 
            for k in range(K)]
    
    p_z[0].eval()
