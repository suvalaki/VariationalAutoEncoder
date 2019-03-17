# testing gmvae

import numpy as np
import tensorflow as tf

# Warn only once
import warnings
warnings.simplefilter(action='ignore')

from importlib import reload  # Python 3.4+ only.


# Check GPU Device

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))


# Get dummary data from Mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

in_dim = x_train.shape[1]
rand_state = 123

# take a subset for testing purposes
x = x_train[0:10000]
y = y_train[0:10000]


# GMVAE 

import gmvae
from gmvae import GMVAE

gmvae = reload(gmvae)
GMVAE = gmvae.GMVAE

model = GMVAE(
    input_dim=in_dim,
    components=10, 
    kind='binary', 
    random_state=rand_state
    )

model.build()

model.fit(
    X_train=x_train, 
    subsample=1000,
    epochs=25,
    iterep=100, 
    y_train=y_train, 
    X_test=x_test, 
    y_test=y_test,
    subsample_test=1000
)

