# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 16:07:53 2019

@author: david
"""

elems = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
alternate = tf.map_fn(lambda x: x[0] * x[1], elems, dtype=tf.int64)
alternate