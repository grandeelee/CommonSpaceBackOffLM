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

#%% reader for toy.set
def _read_words(filename):
  with open(filename, "r") as f:
      return f.read().replace("\n", " <eos> ").split()
  
def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  id_to_word = dict(zip(range(len(words)), words))

  return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]

# requires input data_path
# output train/valid/test_data (length, ), elements in index
def _raw_data(data_path=None):
  train_path = os.path.join(data_path, "train.nltk_tokenizer.txt")
  valid_path = os.path.join(data_path, "valid.nltk_tokenizer.txt")
  test_path = os.path.join(data_path, "test.nltk_tokenizer.txt")
  train_lid_path = os.path.join(data_path, "train.nltk_tokenizer.txt.lid.npy")
  train_lid_weights_path = os.path.join(data_path, "train.nltk_tokenizer.txt.lid_weights.npy")

  word_to_id, id_to_word = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  train_lid = np.load(train_lid_path)
  train_lid_weights = np.load(train_lid_weights_path)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, train_lid, train_lid_weights, valid_data, test_data, vocabulary, id_to_word

# return a queue of input and target [batch_size, nunm_steps]
def sample_producer(raw_data, batch_size, num_steps, train_lid=[], train_lid_weights=[], name=None):
  with tf.name_scope(name, "SampleProducer", [raw_data, train_lid, batch_size, num_steps]):
    if not train_lid == []:
      raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
      train_lid = tf.convert_to_tensor(train_lid, name="train_lid", dtype=tf.int32)
      train_lid_weights = tf.convert_to_tensor(train_lid_weights,
                                               name="train_lid_weights", dtype=tf.float32)

      data_len = tf.size(raw_data)
      batch_len = data_len // batch_size
      data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])
      data_lid = tf.reshape(train_lid[0 : batch_size * batch_len],
                      [batch_size, batch_len])
      data_lid_weights = tf.reshape(train_lid_weights[0: batch_size * batch_len],
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
      y_lid = tf.strided_slice(data_lid, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
      y_lid_weights = tf.strided_slice(data_lid_weights, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
      return x, y, y_lid, y_lid_weights
    if train_lid == []:
      raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
      data_len = tf.size(raw_data)
      batch_len = data_len // batch_size
      data = tf.reshape(raw_data[0: batch_size * batch_len],
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
      return x, y, train_lid, train_lid_weights


# train_data, train_lid, train_lid_weights, valid_data, test_data, vocabulary, id_to_word = _raw_data("data")
#
# x, y, y_lid, y_lid_weights = sample_producer(train_data, 20, 35, train_lid=train_lid, train_lid_weights=train_lid_weights)
# # x, y, y_lid, y_lid_weights = sample_producer(train_data, 20, 35)
# with tf.Session() as sess:
#   coord = tf.train.Coordinator()
#   threads = tf.train.start_queue_runners(coord=coord)
#   for i in range(5):
#     values = sess.run([x, y, y_lid, y_lid_weights])
#   words = [id_to_word[i] for i in values[0][0]]
#   targets = [id_to_word[i] for i in values[1][0]]
#   targets_lid = values[2][0]
#   targets_lid_weights = values[3][0]
#   print(words)
#   print(targets)
#   print(targets_lid)
#   print(targets_lid_weights)
#   coord.request_stop()
#   coord.join(threads)