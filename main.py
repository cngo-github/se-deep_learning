import theano
import logging
import time

import theano.tensor as t
import numpy as np

from corpus import Corpus
from model import Model

import matplotlib.pyplot as plt
plt.ion()

logging.basicConfig(filename = 'log', level = logging.INFO)
logger = logging.getLogger('test')
#logger.setLevel(logging.INFO)

filepath = 'training_test'
vocab = 'vocab_train'

c = Corpus(logger)
c.loadVocabulary(vocab)

n_hid = 50
n_steps = 10
n_seq = 5
n_classes = 3
n_out = n_classes

fs = open(filepath, 'r')
tokens = None

seq, targets, tokens = c.encode(n_seq, n_steps, tokens, fs)
a, b, n_in = seq.shape

t0 = time.time()

params = {
		'n_in': n_in,
		'n_hid': n_hid,
		'n_out': n_out,
		'n_epochs': 250
	}

model = Model(logger, params)

while seq is not None and targets is not None:
	model.fit(seq, targets, validation_freq=1000)

	seq, targets, tokens = c.encode(n_seq, n_steps, tokens, fs)

seqs = xrange(n_seq)

plt.close('all')

for seq_num in seqs:
	fig = plt.figure()
	ax1 = plt.subplot(211)
	plt.plot(seq[seq_num])
	ax1.set_title('input')
	ax2 = plt.subplot(212)

	# blue line will represent true classes
	true_targets = plt.step(xrange(n_steps), targets[seq_num], marker='o')

	# show probabilities (in b/w) output by model
	guess = model.predict_probability(seq[seq_num])
	guessed_probs = plt.imshow(guess.T, interpolation='nearest',
								cmap='gray')
	ax2.set_title('blue: true class, grayscale: probs assigned by model')

print "Elapsed time: %f" % (time.time() - t0)