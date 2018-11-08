#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python back_off_combi.py\
 --gpus=3\
  --inference=True\
   --save_path="tmp/test"\
    --n_clusters=50


@author: grandee
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import os
import numpy as np
import tensorflow as tf
# from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.client import device_lib
import back_off_combi_reader as CSReader
import logging

# <editor-fold desc="logging, FLAGS and CUDA device setup">


flags = tf.flags
flags.DEFINE_string("data_path", "data",
                    "Where the training/test data is stored.")
flags.DEFINE_integer("gpus", "3",
                     "this only run on one gpu "
                     "choose the gpu index to use. ")
flags.DEFINE_bool("inference", False,
                  "Perform inference instead of training")
flags.DEFINE_string("save_path", "tmp/new_model",
                    "Model output directory.")
flags.DEFINE_integer("n_clusters", 700,
                    "choose which cluster embedding to use "
                    "there are 300, 500, 700, 900, 1100, 1300, 1500")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# create file handler which logs even debug messages
fh = logging.FileHandler("log/" + FLAGS.save_path.split("/")[-1])
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)
logging = tf.logging

# only use one GPU in this case 3, appear to tf as GPU:0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpus)
print(device_lib.list_local_devices())


# </editor-fold>

# input: config, raw_data
class InputData(object):
	"""The input data."""

	def __init__(self, config, data, data_class, name=None):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
		# input shape is [batch_size, num_steps]
		self.input_data, self.targets, self.class_t, self.class_t1 = CSReader.sample_producer(
			data, data_class, batch_size, num_steps, name=name)


# model
class CSLModel(object):
	"""The language model"""

	def __init__(self, is_training, config, input_):
		self._is_training = is_training
		self._input = input_
		self.target = input_.targets
		self.batch_size = input_.batch_size
		self.num_steps = input_.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size


		with tf.variable_scope("Class_Net"):
			with tf.device("/cpu:0"):
				# load pretrained embedding from vanilla language model
				pretrained_embeddings = np.load("data/model_300_unk_embedding.npy")
				# load embedding cluster derived from kmean of pretrained embedding
				class_embedding = np.load("data/monolingual_embedding/model_300_unk_embedding.npy_50c_embed.npy")
				# do normalization
				class_norm = np.linalg.norm(class_embedding, axis=1)
				word_norm = np.linalg.norm(pretrained_embeddings, axis=1)
				class_norm = np.reshape(class_norm, (-1,1))
				word_norm = np.reshape(word_norm, (-1,1))
				class_embedding = np.divide(class_embedding, class_norm)
				pretrained_embeddings = np.divide(pretrained_embeddings, word_norm)
				# cluster embedding
				cluster_embedding = tf.get_variable(
					"class_embedding",
					[FLAGS.n_clusters, size],
					dtype=tf.float32,
					initializer=tf.constant_initializer(class_embedding, verify_shape=True),
					trainable=False
				)
				# shape [batch_size, num_steps, hidden_dim]
				cluster_inputs = tf.nn.embedding_lookup(cluster_embedding, input_.class_t)
				# class_target = tf.nn.embedding_lookup(cluster_embedding, input_.class_t1)

			def make_cell_class():
				cell = tf.contrib.rnn.LSTMBlockCell(
					300, forget_bias=0.0)
				if is_training and config.keep_prob < 1:
					cell = tf.contrib.rnn.DropoutWrapper(
						cell, output_keep_prob=config.keep_prob)
				return cell

			# no dropout during reference
			if is_training and config.keep_prob < 1:
				cluster_inputs = tf.nn.dropout(cluster_inputs, config.keep_prob)

			# input is cluster_inputs
			cluster_cell = tf.contrib.rnn.MultiRNNCell(
		        [make_cell_class() for _ in range(config.num_layers)],
		        state_is_tuple=True)

			self._initial_state_cluster = cluster_cell.zero_state(config.batch_size, tf.float32)
			state_class = self._initial_state_cluster
			cluster_inputs_unstacked = tf.unstack(cluster_inputs, num=config.num_steps, axis=1)
			outputs_class, state_class = tf.contrib.rnn.static_rnn(cluster_cell,
		                                               cluster_inputs_unstacked, initial_state=state_class)
			self._final_state_cluster = state_class
			# this cluster output will be concatenated with word output
			cluster_output = tf.reshape(tf.concat(outputs_class, 1), shape=(-1, config.hidden_size))

			softmax_w_class = tf.get_variable(
				"softmax_w_class",
				[300, FLAGS.n_clusters],
				dtype=tf.float32,
				initializer=tf.constant_initializer(np.transpose(class_embedding), verify_shape=True),
				trainable=False
			)
			softmax_b_class = tf.get_variable(
				"softmax_b_class",
				[FLAGS.n_clusters],
				dtype=tf.float32,
				initializer=tf.zeros_initializer,
				trainable=False)
			logits_class = tf.nn.xw_plus_b(cluster_output, softmax_w_class, softmax_b_class)
			logits_class = tf.reshape(logits_class, [config.batch_size, config.num_steps, FLAGS.n_clusters])
			loss_class = tf.contrib.seq2seq.sequence_loss(
			# loss_class = tf.nn.sigmoid_cross_entropy_with_logits(
				logits_class,
				input_.class_t1,
				tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
				average_across_timesteps=False,
				average_across_batch=True)
			self._cost_class = tf.reduce_sum(loss_class)
			# self._cost_class = tf.minimum(tf.reduce_sum(tf.reduce_mean(loss_class, reduction_indices=0)), 10)

		with tf.variable_scope("Word_Net"):
			word_embedding = tf.get_variable(
				"word_embedding",
				[vocab_size, size],
				dtype=tf.float32,
				initializer=tf.constant_initializer(pretrained_embeddings, verify_shape=True),
				trainable=False
			)
			# concat input with class output
			embedding_inputs = tf.nn.embedding_lookup(word_embedding, input_.input_data)
			with tf.control_dependencies([embedding_inputs, cluster_output]):
				cluster_output = tf.reshape(cluster_output, [config.batch_size, config.num_steps, size])
				# cluster_output = tf.multiply(tf.add(cluster_output, -1.7), 3.4)
				inputs = tf.concat([embedding_inputs, cluster_output], 2)
				print("the input shape after concatenation is ", inputs.get_shape().as_list())

			def make_cell():
				cell = tf.contrib.rnn.LSTMBlockCell(
					600, forget_bias=0.0)
				if is_training and config.keep_prob < 1:
					cell = tf.contrib.rnn.DropoutWrapper(
						cell, output_keep_prob=config.keep_prob)
				return cell
			# this rnn is for word_embedding
			cell = tf.contrib.rnn.MultiRNNCell(
				[make_cell() for _ in range(config.num_layers)],
				state_is_tuple=True)

			self._initial_state = cell.zero_state(config.batch_size, tf.float32)
			state = self._initial_state
			# slice input along axis 1
			inputs = tf.unstack(inputs, num=config.num_steps, axis=1)
			outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=state)
			# word_weights of dimension [batch_size, num_steps, hidden_dim]
			output = tf.reshape(tf.concat(outputs, 1), shape=(-1, 600))

			softmax_w = tf.get_variable(
				"softmax_w",
				[600, config.vocab_size],
				dtype=tf.float32)
			softmax_b = tf.get_variable(
				"softmax_b",
				[config.vocab_size],
				dtype=tf.float32)
			logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
			logits = tf.reshape(logits, [config.batch_size, config.num_steps, config.vocab_size])
			loss = tf.contrib.seq2seq.sequence_loss(
				logits,
				input_.targets,
				tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
				average_across_timesteps=False,
				average_across_batch=True)
			self._cost = tf.reduce_sum(loss)
			self._final_state = state
		if not is_training:
			return

		# learning rate is set in run_epoch
		self._lr = tf.Variable(0.0, trainable=False)
		# a list of values is returned
		tvars_class = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Model/Class_Net")
		tvars_word = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Model/Word_Net")
		tvars = tvars_word + tvars_class
		print([v.name for v in tvars])
		# a list of tensor is returned, so []+[]
		grads_class, _ = tf.clip_by_global_norm(tf.gradients(self._cost_class, tvars_class),
		                                  config.max_grad_norm)
		grads_word, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars_word),
		                                       config.max_grad_norm)
		grads = grads_word + grads_class
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(
			zip(grads, tvars),
			global_step=tf.train.get_or_create_global_step())

		self._new_lr = tf.placeholder(
			tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	@property
	def input(self):
		return self._input

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def initial_state_class(self):
		return self._initial_state_cluster

	@property
	def cost(self):
		return self._cost

	@property
	def cost_class(self):
		return self._cost_class

	@property
	def final_state(self):
		return self._final_state

	@property
	def final_state_class(self):
		return self._final_state_cluster

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op


# fetch all ops and feed final state as initialization for next state
# compute perplexity every one tenth of training data
def run_epoch(session, model, id_to_word, eval_op=None, verbose=False, istest=False):
	"""Runs the model on the given data."""
	start_time = time.time()
	costs = 0.0
	costs_class = 0.0
	iters = 0
	state, state_class = session.run([model.initial_state, model.initial_state_class])
	fetches = {
		"cost": model.cost,
		"cost_class": model.cost_class,
		"final_state": model.final_state,
		"final_state_class":model.final_state_class,
		"word": model.target,
	}
	if eval_op is not None:
		fetches["eval_op"] = eval_op
	# for printing words
	test_sentence = []
	test_sentence_entropy = []
	for step in range(model.input.epoch_size):
		feed_dict = {}
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c
			feed_dict[h] = state[i].h
		for i, (c1, h1) in enumerate(model.initial_state_class):
			feed_dict[c1] = state_class[i].c
			feed_dict[h1] = state_class[i].h

		vals = session.run(fetches, feed_dict)
		cost = vals["cost"]
		cost_class = vals["cost_class"]
		# don't delete this, get the previous final state for init
		state = vals["final_state"]
		state_class = vals["final_state_class"]

		if istest:
			wordindex = vals["word"]
			test_sentence_entropy.append(cost)
			test_sentence.append(id_to_word[wordindex[0][0]])
		costs += cost
		costs_class += cost_class
		iters += model.input.num_steps

		if verbose and step % (model.input.epoch_size // 10) == 1:
			logging.info("%.3f class ppl: %.3f perplexity: %.3f speed: %.0f wps" %
			             (step * 1.0 / model.input.epoch_size, np.exp(costs_class / iters), np.exp(costs / iters),
			              iters * model.input.batch_size / (time.time() - start_time)))

	# if this is test then print the two lists to file
	if istest:
		np.save(FLAGS.save_path + "test_sentence", test_sentence)
		np.save(FLAGS.save_path + "test_sentence_entropy", test_sentence_entropy)
	return np.exp(costs / iters)



# Configuration for training and testing, similar to ptb medium
class TestConfig(object):
	"""Tiny config, for testing."""
	init_scale = 0.05
	learning_rate = 1.0
	max_grad_norm = 10
	num_layers = 2
	num_steps = 35
	hidden_size = 300
	max_epoch = 10
	max_max_epoch = 40
	keep_prob = 0.5
	lr_decay = 0.8
	batch_size = 20
	vocab_size = 10000
	rnn_mode = BLOCK


def main(_):
	raw_data = CSReader._raw_data(FLAGS.data_path)
	train_data, train_class, valid_data, valid_class, test_data, test_class, vocab, id_to_word = raw_data
	config = TestConfig()
	config.vocab_size = vocab
	eval_config = TestConfig()
	eval_config.vocab_size = vocab
	eval_config.batch_size = 1
	eval_config.num_steps = 1

	# some path strings here
	# for writer
	writer_path = os.path.join(FLAGS.save_path, "summary")
	save_path = os.path.join(FLAGS.save_path, "model.ckpt")

	is_inference = FLAGS.inference

	tf.reset_default_graph()

	with tf.Graph().as_default() as g:
		initializer = tf.random_uniform_initializer(-config.init_scale,
		                                            config.init_scale)
		# define three separate graphs with shared variable
		with tf.name_scope("Train"):
			train_input = InputData(config=config, data=train_data,
			                        data_class=train_class, name="TrainInput")
			with tf.variable_scope("Model", reuse=None, initializer=initializer):
				mtrain = CSLModel(is_training=True, config=config, input_=train_input)

		with tf.name_scope("Valid"):
			train_input = InputData(config=config, data=valid_data,
			                        data_class=valid_class, name="ValidInput")
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mvalid = CSLModel(is_training=False, config=config, input_=train_input)

		with tf.name_scope("Test"):
			train_input = InputData(config=eval_config, data=test_data,
			                        data_class=test_class, name="TestInput")
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mtest = CSLModel(is_training=False, config=eval_config, input_=train_input)

		saver = tf.train.Saver()
		with tf.Session() as sess:
			# for dequeue function in InputData()
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			# for tensorboard
			writer = tf.summary.FileWriter(writer_path, sess.graph)
			# init whatever
			init = tf.global_variables_initializer()
			sess.run(init)
			# training stage and save
			if not is_inference:
				for i in range(config.max_max_epoch):
					# try adding a max(decay, 0.002)
					lr_decay = max(config.lr_decay ** max(i + 1 - config.max_epoch, 0.0), 0.002)
					mtrain.assign_lr(sess, config.learning_rate * lr_decay)

					logging.info("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(mtrain.lr)))
					train_perplexity = run_epoch(sess, mtrain, id_to_word,
					                             eval_op=mtrain.train_op, verbose=True, istest=False)
					logging.info("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

					valid_perplexity = run_epoch(sess, mvalid, id_to_word, istest=True)
					logging.info("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

					if i % 3 == 2:
						# save using global steps
						var = g.get_tensor_by_name("Model/global_step:0")
						global_step = sess.run(var)
						save_path_confirm = saver.save(sess, save_path, global_step=global_step)
						logging.info("Model saved in path: %s" % save_path_confirm)

				test_perplexity = run_epoch(sess, mtest, id_to_word, istest=True)
				logging.info("Test Perplexity: %.3f" % test_perplexity)

				# save using global steps
				var = g.get_tensor_by_name("Model/global_step:0")
				global_step = sess.run(var)
				save_path_confirm = saver.save(sess, save_path, global_step=global_step)
				logging.info("Model saved in path: %s" % save_path_confirm)

			# based on the same model, load the saved variables
			if is_inference:
				saver.restore(sess, "tmp/backoff_combi_50/model.ckpt-59640")

				# valid_perplexity = run_epoch(sess, mvalid)
				# print("Valid Perplexity: %.3f" % (valid_perplexity))

				test_perplexity = run_epoch(sess, mtest, id_to_word, istest=True)
				logging.info("Test Perplexity: %.3f" % test_perplexity)
				# var1 = g.get_tensor_by_name("Model/Embedding/embedding:0")
				# embedding = sess.run(var1)
			# np.save(FLAGS.save_path + "embedding", embedding)
			# decode_text(sess, mtest, id_to_word=id_to_word)

			# releaes resources
			coord.request_stop()
			coord.join(threads)


if __name__ == "__main__":
	tf.app.run()

