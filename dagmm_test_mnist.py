# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 11:47:38 2019

@author: david
"""

import tensorflow as tf
import numpy as np
from dagmm import DAGMM 

import mnist
#mnist.init()

x_train, t_train, x_test, t_test = mnist.load()


# Initialize
model = DAGMM(
  comp_hiddens=[2000,500,500,64,2], comp_activation=tf.nn.tanh,
  est_hiddens=[100,50,10], est_activation=tf.nn.tanh,
  est_dropout_ratio=0, epoch_size=5000
)

# Fit the training data to model
x_train_sample = x_train[np.random.choice(a=x_train.shape[1], size=5000), :]
model.fit(x_train)


# Get tensorflwo nodes
[n for n in tf.contrib.graph_editor.get_tensors(model.graph)][1:250]
# Get important values
x_train_sample_reshape = np.reshape(x_train_sample, (-1,) + x_train_sample.shape )

z_c = 'CompNet/Encoder/layer_3/BiasAdd:0'
gamma = 'EstNet/softmax/Reshape_1:0'


z_c_out, gamma_out = model.sess.run([z_c, gamma], feed_dict={'input:0':x_train_sample})

class_out = np.argmax(gamma_out, axis = 1)
unique, counts = np.unique(class_out, return_counts=True)
print(dict(zip(unique, counts)))


from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns


tsne = TSNE()
tsne_z_c = tsne.fit_transform(z_c_out)
df = pd.DataFrame({'z1':tsne_z_c[:,0], 'z2':tsne_z_c[:,1], 'cat':class_out})

sns.scatterplot(x='z1',y='z2',hue='cat',data=df)


