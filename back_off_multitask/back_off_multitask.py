#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python back_off_multitask.py \
        --gpus=0\
        --inference=False \
        --save_path="tmp/backoff_multitask_1500c_300dim_0.1" \
        --n_clusters="1500" \
        --multitask_weight=0.1

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import os
import numpy as np
import tensorflow as tf
#from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.client import device_lib
import CSReader
import logging


#<editor-fold desc="logging, FLAGS and CUDA device setup">
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
flags.DEFINE_string("n_clusters", "700",
                    "choose which cluster embedding to use "
                    "there are 300, 500, 700, 900, 1100, 1300, 1500")
flags.DEFINE_float("multitask_weight", 0.1,
                   "percentage of multitask cost for backprop")
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
#</editor-fold>

# input: config, raw_data
class InputData(object):
  """The input data."""
  def __init__(self, config, data, lid, weights, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    # input shape is [batch_size, num_steps]
    self.input_data, self.targets, self.lid, self.lid_weights = CSReader.sample_producer(
        data, batch_size, num_steps, train_lid=lid, train_lid_weights=weights, name=name)

# model
class CSLModel(object):
    """The language model"""
    def __init__(self, is_training, config, input_):    
        self._is_training = is_training
        self._input = input_
        self.target = input_.targets
        self.lid = input_.lid
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        
        with tf.device("/cpu:0"):
            # load pretrained embedding from vanilla language model
            pretrained_embeddings = np.load("data/fasttext_train_embed.npy")
            # load embedding cluster derived from kmean of pretrained embedding
            embedding_clusters = np.load("data/embedding_kmean_300dim_{}c.npy".format(FLAGS.n_clusters))
            with tf.variable_scope("Embedding"):
                # word embedding
                word_embedding = tf.get_variable(
                    "embedding",
                    [vocab_size, size],
                    dtype=tf.float32,
                    initializer = tf.constant_initializer(pretrained_embeddings, verify_shape=True),
                    trainable = False
                )
                # shape [batch_size, num_steps, hidden_dim]
                embedding_inputs = tf.nn.embedding_lookup(word_embedding, input_.input_data)
                # cluster embedding
                cluster_embedding = tf.get_variable(
                    "clustering",
                    [vocab_size, size],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(embedding_clusters, verify_shape=True),
                    trainable=False
                )
                # shape [batch_size, num_steps, hidden_dim]
                cluster_inputs = tf.nn.embedding_lookup(cluster_embedding, input_.input_data)
        # no dropout during reference
        with tf.device('/device:GPU:0'):
            if is_training and config.keep_prob < 1:
                embedding_inputs = tf.nn.dropout(embedding_inputs, config.keep_prob)
                cluster_inputs = tf.nn.dropout(cluster_inputs, config.keep_prob)

            def make_cell():
                cell = tf.contrib.rnn.LSTMBlockCell(
                        size, forget_bias=0.0)
                if is_training and config.keep_prob < 1:
                    cell = tf.contrib.rnn.DropoutWrapper(
                            cell, output_keep_prob=config.keep_prob)
                return cell

            # rnn for word weights which will elementwise dot product with word embedding
            # input is embedding_inputs and cluster_inputs
            # ouput is word_weights and cluster_weights
            with tf.variable_scope("RNN_word"):
                # this rnn is for word_embedding
                word_cell = tf.contrib.rnn.MultiRNNCell(
                        [make_cell() for _ in range(config.num_mask_layers)],
                        state_is_tuple=True)
                self._initial_state_word = word_cell.zero_state(config.batch_size, tf.float32)
                state = self._initial_state_word
                # slice input along axis 1
                embedding_inputs_unstack = tf.unstack(embedding_inputs, num=config.num_steps, axis=1)
                outputs, state = tf.contrib.rnn.static_rnn(word_cell,
                                                           embedding_inputs_unstack, initial_state=state)
                # store the final_state for initializing the next epoch with previous state
                self._final_state_word = state
                # word_weights of dimension [batch_size, num_steps, hidden_dim]
                # TO-DO: make sure the activation of RNN is sigmoid
                # the output gate of LSTMBlockCell is sigmoid according to the paper
                # arxiv.org/pdf/1409.2329.pdf
                # so the weigths range 0-1 make sense as "mask"
                word_weights = tf.reshape(tf.concat(outputs, 1),
                                          shape=(config.batch_size, -1,config.hidden_size))
            with tf.variable_scope("RNN_class"):
                # rnn for cluster weigths
                cluster_cell = tf.contrib.rnn.MultiRNNCell(
                    [make_cell() for _ in range(config.num_mask_layers)],
                    state_is_tuple=True)
                self._initial_state_cluster = cluster_cell.zero_state(config.batch_size, tf.float32)
                state = self._initial_state_cluster
                cluster_inputs_unstacked = tf.unstack(cluster_inputs, num=config.num_steps, axis=1)
                outputs, state = tf.contrib.rnn.static_rnn(cluster_cell,
                                                           cluster_inputs_unstacked, initial_state=state)
                self._final_state_cluster = state
                cluster_weights = tf.reshape(tf.concat(outputs, 1),
                                             shape=(config.batch_size, -1, config.hidden_size))
            # here just element wise dot product
            with tf.variable_scope("Weighting"):
                # make sure masks are executed, which should be the case anyway
                with tf.control_dependencies([cluster_weights, word_weights]):
                    input1 = tf.multiply(cluster_weights, cluster_inputs)
                    input2 = tf.multiply(word_weights, embedding_inputs)
                    # shape [batch_size, num_steps, hidden_dim]
                    inputs = input1 + input2
            # the rest is the same
            with tf.variable_scope("RNN2"):
                cell = tf.contrib.rnn.MultiRNNCell(
                    [make_cell() for _ in range(config.num_layers)],
                    state_is_tuple=True)
                self._initial_state = cell.zero_state(config.batch_size, tf.float32)
                state = self._initial_state
                inputs = tf.unstack(inputs, num=config.num_steps, axis=1)
                outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=state)
                output = tf.reshape(tf.concat(outputs, 1), shape=(-1, config.hidden_size))

            with tf.variable_scope("softmax"):
                softmax_w = tf.get_variable(
                      "softmax_w", 
                      [config.hidden_size, config.vocab_size], 
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
            with tf.variable_scope("lid"):
                softmax_w_lid = tf.get_variable(
                    "softmax_w",
                    [config.hidden_size, 2],
                    dtype=tf.float32
                )
                softmax_b_lid = tf.get_variable(
                    "softmax_b",
                    [2],
                    dtype=tf.float32
                )
                logits_lid = tf.nn.xw_plus_b(output, softmax_w_lid, softmax_b_lid)
                logits_lid = tf.reshape(logits_lid, [config.batch_size, config.num_steps, 2])
                loss_lid = tf.contrib.seq2seq.sequence_loss(
                    logits_lid,
                    input_.lid,
                    input_.lid_weights,
                    average_across_timesteps=False,
                    average_across_batch=True)
                self._cost = self._cost * (1.0-FLAGS.multitask_weight)\
                             + tf.reduce_sum(loss_lid) * FLAGS.multitask_weight
        # learning rate is set in run_epoch
        self._lr = tf.Variable(0.0, trainable=False)
        # get all the variables
        tvars = tf.trainable_variables()
        # get gradients
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)
        # SGD
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
        return [self._initial_state, self._initial_state_word, self._initial_state_cluster]

    @property
    def cost(self):
        return self._cost
    
    @property
    def final_state(self):
        return [self._final_state, self._final_state_word, self._final_state_cluster]

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
    iters = 0
    state = session.run(model.initial_state)
    fetches = {
            "cost": model.cost,
            "final_state": model.final_state,
	        "word": model.target,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    # for printing words
    test_sentence = []
    test_sentence_entropy = []
    for step in range(model.input.epoch_size):
        # feed_dict will over-write the initialized zero value with the previous final state
        feed_dict = {}
        for idx, initial_state in enumerate(model.initial_state):
            # c and h contain the model name string
            for i, (c, h) in enumerate(initial_state):
                feed_dict[c] = state[idx][i].c
                feed_dict[h] = state[idx][i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        # don't delete this, get the previous final state for init
        state = vals["final_state"]
        if istest:
            wordindex = vals["word"]
            test_sentence_entropy.append(cost)
            test_sentence.append(id_to_word[wordindex[0][0]])
        costs += cost
        iters += model.input.num_steps
            
        if verbose and step % (model.input.epoch_size // 10) == 1:
            logging.info("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size/(time.time() - start_time)))

    # if this is test then print the two lists to file
    if istest:
        sent_path = os.path.join(FLAGS.save_path, "test_sentence")
        entropy_path = os.path.join(FLAGS.save_path, "test_sentence_entropy",)
        np.save(sent_path, test_sentence)
        np.save(entropy_path, test_sentence_entropy)
    return np.exp(costs / iters)

# decoder for inference and text generation
def decode_text(session, model, iters=35, id_to_word=None):
    # decoding
    start_token = [[0]]
    state = session.run(model.initial_state)
    # fetch softmax output
    softmax_tensor = session.graph.get_tensor_by_name("Test/Model/softmax/xw_plus_b:0")[0]
    fetches = {
            "final_state": model.final_state,
            "softmax_out": softmax_tensor 
    }
    sentence = []
    for i in range(iters):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        # feed start_token 
        feed_dict["Test/TestInput/StridedSlice:0"] = start_token
        # run sess on mtest
        vals = session.run(fetches, feed_dict)
        softmax_logit = vals["softmax_out"]
        # get argmax of softmax output
        softmax_argmax = np.argmax(softmax_logit)
        # update new start_token
        start_token = [[softmax_argmax]]
        # update feed
        state = vals["final_state"]
        sentence.append(id_to_word[softmax_argmax])
    print(sentence)

# Configuration for training and testing, similar to ptb medium
class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_mask_layers = 2
    num_steps = 35
    hidden_size = 300
    # based on observation increase from 6 to 10
    max_epoch = 20
    # change from 39 to 50
    max_max_epoch = 55
    keep_prob = 0.35
    # chenge from 0.8 to 0.95 to delay less
    lr_decay = 1/1.15
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK

def main(_):

    raw_data = CSReader._raw_data(FLAGS.data_path)
    train_data, train_data_lid, train_data_lid_weights, valid_data, test_data, vocab, id_to_word = raw_data
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

    is_inference  = FLAGS.inference

    tf.reset_default_graph()

    with tf.Graph().as_default() as g:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        # define three separate graphs with shared variable
        with tf.name_scope("Train"):
            train_input = InputData(config=config, data=train_data,
                                    lid=train_data_lid, weights=train_data_lid_weights, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                mtrain = CSLModel(is_training=True, config=config, input_=train_input)

        with tf.name_scope("Valid"):
            train_input = InputData(config=config, data=valid_data,
                                    lid=[], weights=[], name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = CSLModel(is_training=False, config=config, input_=train_input)

        with tf.name_scope("Test"):
            train_input = InputData(config=eval_config, data=test_data,
                                    lid=[], weights=[], name="TestInput")
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

                    valid_perplexity = run_epoch(sess, mvalid, id_to_word)
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
                saver.restore(sess, "tmp/model_vanilla/model.ckpt-58149")

                # valid_perplexity = run_epoch(sess, mvalid)
                # print("Valid Perplexity: %.3f" % (valid_perplexity))

                test_perplexity = run_epoch(sess, mtest, id_to_word, istest=True)
                logging.info("Test Perplexity: %.3f" % test_perplexity)
                var1 = g.get_tensor_by_name("Model/Embedding/embedding:0")
                embedding = sess.run(var1)
                # np.save(FLAGS.save_path + "embedding", embedding)
                # decode_text(sess, mtest, id_to_word=id_to_word)

            # releaes resources
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
  tf.app.run()
     
