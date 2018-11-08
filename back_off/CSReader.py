#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:05:10 2018

@author: grandee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf
import numpy as np



# requires input data_path
# output train/valid/test_data (length, ), elements in index
def _raw_data(data_path=None):
  train_path = os.path.join(data_path, "model_300_unk.train.nltk_tokenizer.txt.100class.npy")
  valid_path = os.path.join(data_path, "model_300_unk.valid.nltk_tokenizer.txt.100class.npy")
  test_path = os.path.join(data_path, "model_300_unk.test.nltk_tokenizer.txt.100class.npy")

  train_data = np.load(train_path)
  valid_data = np.load(valid_path)
  test_data = np.load(test_path)

  vocabulary = 100
  return train_data, valid_data, test_data, vocabulary

# return a queue of input and target [batch_size, nunm_steps]
def sample_producer(raw_data, batch_size, num_steps, name=None):
  with tf.name_scope(name, "SampleProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])
    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y


