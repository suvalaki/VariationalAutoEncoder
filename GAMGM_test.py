# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 02:53:26 2019

@author: david
"""

from GAMGM import *


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

model = DAGMM()
model.build_model(784,64,10,2)

mnist.train.next_batch(100)[0].shape

model.sess.run(model.minimizer, 
               feed_dict={'x:0': mnist.train.next_batch(1000)[0]})


model.train( mnist, epoch_size=1000, n_batch=1000)


variables_to_query = [
        'estimation_network/estimation_network/prob:0',
        model.latent,
        ]

query = model.sess.run(variables_to_query, 
               feed_dict={'x:0': mnist.train.next_batch(1)[0]})[0]

query

probs = model.sess.run('estimation_network/estimation_network/prob:0', 
               feed_dict={'x:0': mnist.train.next_batch(1)[0]})[0]

group = [np.argmax(query[0][i]) for i in range(len(query[0])) ]

[n.name for n in tf.get_default_graph().as_graph_def().node][1:200]

import seaborn as sns

sns.scatterplot(x=query[1].T[0],y=query[1].T[2],hue=group)




import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

data, _ = make_blobs(n_samples=1000, n_features=5, centers=5, random_state=123)
data[300] = [-1, -1, -1, -1, -1]
data[500] = [ 1,  0,  1,  1,  1]
ano_index = [300, 500]

plt.figure(figsize=[8,8])
plt.plot(data[:,0], data[:,1], ".")
plt.plot(data[ano_index,0], data[ano_index,1], "o", c="r", markersize=10)

model = DAGMM()
model.build_model(5,5,3,2)
model.train( mnist, epoch_size=10000, n_batch=1000)

