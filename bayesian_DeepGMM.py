# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 16:20:13 2019

@author: david
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


# Parameters
n_features = 786

def graph_Qmu_sigma_g_xin(xin):
    
    


with tf.Graph().as_default() as graph:
    
    input = tf.placeholder(dtype=tf.float32, 
                           shape=[None, n_features], name='input')
    
    with tf.variable_scope('Encoder'):
        