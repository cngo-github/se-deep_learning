import theano
import logging
import time

import theano.tensor as t
import numpy as np

from corpus import Corpus
from model import Model

logfile = 'log_' + str(time.time())

logging.basicConfig(filename = logfile, level = logging.INFO)
logger = logging.getLogger('test')

filepath = 'train1K'
vocab = 'vocab_train1K'

c = Corpus(logger)
c.loadVocabulary(vocab)

n_hid = 50
n_steps = 10
n_seq = 5
n_classes = 3
n_out = n_classes

fs = open(filepath, 'r')
tokens = None

softmax_time = 0

#Retrives the inputs and targets
seq, targets, tokens = c.encode(n_seq, n_steps, tokens, fs)
_, __, n_in = seq.shape

t0 = time.time()

#Creates the model to run the RNN.
params = {
		'n_in': n_in,
		'n_hid': n_hid,
		'n_out': n_out,
		'n_epochs': 250
	}

model = Model(logger, params)

#Trains the RNN and runs the softmax signal.
while seq is not None and targets is not None:
	model.fit(seq, targets, validation_freq=1000)

	seqs = xrange(n_seq)
	for seq_num in seqs:
		tsm = time.time()
		guess = model.predict_probability(seq[seq_num])

		tsm = time.time() - tsm
		softmax_time += tsm
		logger.info("Softmax elapsed time: %f" % (tsm))

	seq, targets, tokens = c.encode(n_seq, n_steps, tokens, fs)

logger.info("Total elapsed time: {} and softmax time: {} ".format(time.time() - t0, softmax_time))