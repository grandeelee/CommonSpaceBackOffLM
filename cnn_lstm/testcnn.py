#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:37:54 2018

@author: grandee
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.reset_default_graph()
with tf.Graph().as_default() as g:
    x = tf.random_normal([20,30,300,1])
    print("input shape is ", x.get_shape())
    conv1 = tf.layers.conv2d(
          inputs=x,
          filters=100,
          kernel_size=[1, 300],
          padding="valid",
          use_bias=False,
          activation=tf.nn.relu)
    print("input shape is ", conv1.get_shape())
    conv1 = tf.reshape(conv1, [-1,100])
    print("output shape is ", conv1.get_shape())
    print(tf.trainable_variables())
    
    word_kernel = g.get_tensor_by_name('conv2d/kernel:0')
    print("kernel shape is ", word_kernel.get_shape())
    word_kernel = tf.reshape(word_kernel, [300,100])
    print("kernel shape is ", word_kernel.get_shape())
    
    result = tf.matmul(conv1, word_kernel, transpose_b=True)
    result = tf.reshape(result, [20,30,-1])
    print("result shape is ", result.get_shape())

