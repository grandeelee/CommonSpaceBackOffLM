#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 22:39:53 2018

@author: grandee
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


def main(_):
	embeddings = np.load('crosslingual_LM/data/embedding.npy')
	with open("crosslingual_LM/data/pseudo_cs_data_small_with_mono_vocab", "r") as infile:
		words = infile.read().split("\n")

	with tf.variable_scope("Embedding"):
		embedding = tf.get_variable(
			"embedding",
			[len(words), 300],
			dtype=tf.float32)
		embedding = tf.assign(embedding, embeddings)
		saver = tf.train.Saver()
	with tf.Session() as sess:
		writer = tf.summary.FileWriter("expt_embed_dist/tmp/embed_visual/summary/", sess.graph)
		sess.run(embedding)
		saver.save(sess, "expt_embed_dist/tmp/embed_visual/model.ckpt")
		writer.close()


if __name__ == "__main__":
	tf.app.run()