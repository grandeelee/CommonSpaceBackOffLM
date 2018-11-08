#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:41:57 2018

@author: grandee
"""
#%% imported goods
import collections
import os
import argparse
import keras
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical #, plot_model
from keras import optimizers

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

#%% Argparser
parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, help="rnn time-steps")
parser.add_argument("--hidden_size", type=int, help="LSTM hidden layer size")
parser.add_argument("--dropout", type=float, help="dropout layer for LSTM")
parser.add_argument("--num_epochs", type=int, help="number of epochs to run")
# set default values for parser
parser.set_defaults(num_steps=35, hidden_size=1500, dropout=0.5, num_epochs=10)
args = parser.parse_args()


#%% Reader
def _read_words(filename):
    with open(filename) as f:
        return f.read().replace("\n", " <eos> ").split(' ')

def _build_vocab(filename):
    """Return word_to_id mapping sorted according to frequency"""
    
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def raw_data(data_path=None):
    """Load raw data from data directory "data_path".
    Args:
    data_path: string path to the directory 
    Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    """

    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")
    test_path = os.path.join(data_path, "test.txt")

    # build word id from train txt
    word_to_id = _build_vocab(train_path)
    # convert train txt to int32
    train_data = _file_to_word_ids(train_path, word_to_id)
    # convert valid txt
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    # convert test txt
    test_data = _file_to_word_ids(test_path, word_to_id)
    # vocab length for the config 
    vocabulary = len(word_to_id)
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    return train_data, valid_data, test_data, vocabulary, id_to_word


#%% Generator
class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=35):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step
        
    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
                temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step
            yield x, y
            
            
#%% Prepare data 
num_steps = args.num_steps
hidden_size = args.hidden_size
DROPOUT = args.dropout
num_epochs = args.num_epochs
batch_size = 20
# get raw data, each data is a list of tokens
train_data, valid_data, test_data, vocabulary, id_to_word= raw_data('.')

# calculation for the fit_generator function
train_steps = np.size(train_data)//(batch_size*num_steps)
validation_steps = np.size(valid_data)//(batch_size*num_steps)
test_steps = np.size(test_data)//(batch_size*num_steps)
# get generator, each sample in shape of [batch_size, num_steps] 
train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)
# for speed, in real case change num_steps to len(test_data)-1, batch_size=1
test_data_generator = KerasBatchGenerator(test_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)


#%% Model
inputs = Input(shape=(None,))
lookup_layer = Embedding(vocabulary, hidden_size, input_length=None)(inputs)
lstm_1 = LSTM(hidden_size, return_sequences=True, activation='relu', 
              input_shape=(None, hidden_size))(lookup_layer)
lstm_2 = LSTM(hidden_size, return_sequences=True, activation='relu')(lstm_1)
dropout_layer = Dropout(DROPOUT)(lstm_2)
output = TimeDistributed(Dense(vocabulary), input_shape=(None, hidden_size))(lstm_2)
softmax_out = Activation('softmax')(output)
model = Model(inputs=inputs, outputs=softmax_out)

sgd = optimizers.SGD(lr=1.0, clipnorm=5.0, momentum=0.0, decay=0.8, nesterov=False)
adadelta = optimizers.Adadelta(lr=1.0, clipnorm=5.0, rho=0.95, epsilon=None, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=adadelta)
# save a pic, a json
# plot_model(model, to_file='data/model.png')
json_string = model.to_json()
with open('data/model_json', 'w') as outfile:
    outfile.write(json_string)

#%% Training
checkpointer = ModelCheckpoint(filepath='data' + '/model-{epoch:02d}.hdf5', verbose=1)

for run_num in range(num_epochs):
    print('Epoch {}: \n'.format(run_num+1))
    train_loss = model.fit_generator(train_data_generator.generate(), train_steps, 
                        epochs=1, verbose=1,validation_data=valid_data_generator.generate(),
                        validation_steps=validation_steps, callbacks=[checkpointer])
    print('Train perplexity = {}'.format(np.exp(train_loss.history['loss'])))
    valid_loss = model.evaluate_generator(valid_data_generator.generate(), validation_steps)
    print('Valid perplexity = {}'.format(np.exp(valid_loss)))
    test_loss = model.evaluate_generator(test_data_generator.generate(), test_steps)
    print('Test perplexity = {}'.format(np.exp(test_loss)))

