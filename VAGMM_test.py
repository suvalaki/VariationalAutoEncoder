# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 01:10:43 2019

@author: david
"""


import tensorflow as tf
import numpy as np
from VAGMM import *
import mnist
#mnist.init()

x_train, t_train, x_test, t_test = mnist.load()

session = graph_build()