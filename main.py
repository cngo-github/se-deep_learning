from corpus import Corpus
from model import Model

import numpy as np

import logging
import time
#import theano
#import rnn

def main(vocabFile, testFile):
	c = Corpus(vocabFile)
	seq = c.encodeAllTokens(testFile)
	inSize = c.getVocabSize()

	run_softmax(seq, seq, n_hidden = 10, n_in = inSize, n_steps = 10,
			n_seq = 100, n_classes = 1, n_out = inSize)


def run_softmax(seq, targets, n_hidden, n_in, n_steps, n_seq, n_classes, n_out,
		n_epochs = 250):
	seq = np.matrix(seq)
	targets = np.asarray(targets)

	model = Model(logger = logger, n_in=n_in, n_hidden=n_hidden, n_out=n_out,
			learning_rate=0.001, learning_rate_decay=0.999,
			n_epochs=n_epochs, activation='sigmoid')

	model.fit(seq, targets, validation_frequency=1000)
#c = corpus.Corpus('test')
#print(c.encodeAllTokens('test'))

#print(c.getVocabSize())
#for lst in c.classes:
#	if not lst:
#		print("empty at index: ", i)
#print(c.encodedTokens)
'''
n_input, n_hidden, n_output, n_class, learning_rate = 0.01, n_epochs = 100, activation = 'sigmoid'

aModel = Model({n_input: 5,
		n_hidden: 10,
		n_output: 10,
		n_class: 10})

aModel.fit(seq, targets)
'''

logger = logging.getLogger(__name__)

main('test', 'test2')
