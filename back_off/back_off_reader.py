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


def _read_words(filename):
  with open(filename, "r") as f:
      return f.read().replace("\n", " </s> ").split()

def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  # always write the vocab list to dist
  with open(filename + ".vocab", "w") as f:
    f.writelines("\n".join(word for word in words))
  word_to_id = dict(zip(words, range(len(words))))
  id_to_word = dict(zip(range(len(words)), words))
  return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] if word in word_to_id else word_to_id["<unk>"] for word in data]

# requires input data_path
# output train/valid/test_data (length, ), elements in index
def _raw_data(data_path=None):
  train_path = os.path.join(data_path, "model_300_unk.train.nltk_tokenizer.txt.100class.npy")
  valid_path = os.path.join(data_path, "model_300_unk.valid.nltk_tokenizer.txt.100class.npy")
  test_path = os.path.join(data_path, "model_300_unk.test.nltk_tokenizer.txt.100class.npy")

  train_class = np.load(train_path)
  valid_class = np.load(valid_path)
  test_class = np.load(test_path)

  train_path = os.path.join(data_path, "train.nltk_tokenizer.txt")
  valid_path = os.path.join(data_path, "valid.nltk_tokenizer.txt")
  test_path = os.path.join(data_path, "test.nltk_tokenizer.txt")

  word_to_id, id_to_word = _build_vocab(train_path)

  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)

  vocabulary = len(word_to_id)
  return train_data, train_class, valid_data, valid_class, test_data, test_class, vocabulary, id_to_word

# return a queue of input and target [batch_size, nunm_steps]
def sample_producer(raw_data, raw_class, batch_size, num_steps, name=None):
  with tf.name_scope(name, "SampleProducer", [raw_data, raw_class, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    raw_class = tf.convert_to_tensor(raw_class, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size
    data = tf.reshape(raw_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])
    data_class = tf.reshape(raw_class[0 : batch_size * batch_len],
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
    class_tplus1 = tf.strided_slice(data_class, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    class_tplus1.set_shape([batch_size, num_steps])
    return x, y, class_tplus1


