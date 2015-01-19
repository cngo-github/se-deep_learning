from rnn import RNN

import theano.tensor as t
import numpy as np
import cPickle as pickle

import theano
import uuid
import logging
import time

class Model(object):
	def __init__(self, params):
		self.setparams(params)
		self.ready()

	def setparams(self, params):
		if not params.has_key('samples'):
			self.samples = 1
		else:
			self.samples = params['samples']

		if not params.has_key('batch'):
			self.batch = 1
		else:
			self.batch = params['batch']

		if not params.has_key('seqlen'):
			self.seqlen = 1
		else:
			self.seqlen = params['seqlen']

		if not params.has_key('activation'):
			self.activation = t.nnet.sigmoid
		else:
			self.activation = params['activation']

		if not params.has_key('output_type'):
			self.output_type = t.nnet.softmax
		else:
			self.output_type = params['output_type']

		if not params.has_key('rnn_dtype'):
			self.rnn_dtype = theano.config.floatX
		else:
			self.rnn_dtype = params['rnn_dtype']

		if not params.has_key('learning_rate'):
			self.learning_rate = 0.01
		else:
			self.learning_rate = float(params['learning_rate'])

		if not params.has_key('n_epochs'):
			self.n_epochs = 1000
		else:
			self.n_epochs = int(params['n_epochs'])

		# RNN parameters
		if not params.has_key('n_in'):
			self.n_in = 5
		else:
			self.n_in = int(params['n_in'])

		if not params.has_key('n_hid'):
			self.n_hid = 50
		else:
			self.n_hid = int(params['n_hid'])

		if not params.has_key('n_out'):
			self.n_out = 5
		else:
			self.n_out = int(params['n_out'])

		self.ready()

	def getparams(self):
		params = {
			'n_in': self.n_in,
			'n_hid': self.n_hid,
			'n_out': self.n_out,
			'learning_rate': self.learning_rate,
			'n_epochs': self.n_epochs,
			'rnn_dtype': self.rnn_dtype,
			'activation': self.activation,
			'output_type': self.output_type,
			'samples': self.samples,
			'batch': self.batch,
			'seqlen': self.seqlen
		}

		return params

	def ready(self):
		params = {
			'n_in': self.n_in,
			'n_hid': self.n_hid,
			'n_out': self.n_out,
			'activation': self.activation,
			'output_type': self.output_type,
			'dtype': self.rnn_dtype
		}

		self.rnn = RNN(params)

	def fit(self, inputs, target):
		shared_x, shared_y = self.share_datasets((inputs, target))
		trainfn = self.rnn.gettrainfn(shared_x, shared_y, self.learning_rate)

		inIdx = [i * self.seqlen * self.batch for i in xrange(self.samples + 1)]
		tgtIdx = inIdx
		seq = {
			'inputs': inIdx,
			'targets': tgtIdx
		}

		start_time = time.clock()
		logging.info('Running ({} epochs)'.format(self.n_epochs))

		for _ in xrange(self.n_epochs):
			for i in range(0, self.samples, self.batch):
				trainfn(seq['inputs'][i], seq['inputs'][i + self.batch], seq['targets'][i], seq['targets'][i + self.batch])
		logging.info('===== training epoch time (%.5fm)' % ((time.clock() - start_time) / 60))

#	def setpredictfn(self):
#		self.predict_probability = theano.function(inputs = [self.x, ],
#										outputs = self.rnn.prob_y)
#		self.predict = theano.function(inputs = [self.x, ],
#							outputs = self.rnn.y_out)

	def __getstate__(self):
		params = self.getparams()
		weights = self.rnn.getweights()

		return (params, weights)

	def __setstate__(self, state):
		params, weights = state

		self.setparams(params)
		self.rnn.setweights(weights)

	def load(self, path):
		fs = open(path, 'rb')
		
		logging.info("Model state loading from file " + path)
		state = pickle.load(fs)
		self.__setstate__(state)
		
		fs.close()
		logging.info("Model state loaded.")

	def save(self, path = None):
		if not path:
			path = str(uuid.uuid4())

		fs = open(path, 'wb')
		state = self.__getstate__()
		pickle.dump(state, fs, protocol = pickle.HIGHEST_PROTOCOL)
		fs.close()

		logging.info("Model state saved to file " + path)

#		self.predict_proba = theano.function(inputs=[self.x, ],
#											outputs=self.rnn.prob_y)
#		self.predict = theano.function(inputs=[self.x, ],
#									outputs=self.rnn.y_out)

	def share_datasets(self, data_xy):
		""" Load the dataset into shared variables """

		data_x, data_y = data_xy
		
		shared_x = theano.shared(np.asarray(data_x, dtype= theano.config.floatX))
		shared_y = theano.shared(np.asarray(data_y, dtype= theano.config.floatX))

		return shared_x, t.cast(shared_y, 'int32')
'''
	def fit(self, X_train, Y_train, X_test=None, Y_test=None,
			validation_frequency=100):
		""" Fit model

		Pass in X_test, Y_test to compute test error and report during
		training.

		X_train : ndarray (n_seq x n_steps x n_in)
		Y_train : ndarray (n_seq x n_steps x n_out)

		validation_frequency : int
		in terms of number of sequences (or number of weight updates)
		"""
		if X_test is not None:
			assert(Y_test is not None)
			self.interactive = True
			test_set_x, test_set_y = self.shared_dataset((X_test, Y_test))
		else:
			self.interactive = False

		train_set_x, train_set_y = self.share_datasets((X_train, Y_train))
		n_train = train_set_x.get_value(borrow=True).shape[0]

#		if self.interactive:
#			n_test = test_set_x.get_value(borrow=True).shape[0]

		######################
		# BUILD ACTUAL MODEL #
		######################
#		if self.logger is not None:
#			self.logger.info('... building the model')

		index = t.lscalar('index')    # index to a case
		# learning rate (may change)
		l_r = t.scalar('l_r', dtype=theano.config.floatX)
#		mom = T.scalar('mom', dtype=theano.config.floatX)  # momentum

		cost = self.rnn.loss(self.y)

		compute_train_error = theano.function(inputs=[index, ],
						outputs=self.rnn.loss(self.y),
						givens={
							self.x: train_set_x[index],
							self.y: train_set_y[index]})

#		if self.interactive:
#			compute_test_error = theano.function(inputs=[index, ],
#					outputs=self.rnn.loss(self.y),
#					givens={
#						self.x: test_set_x[index],
#						self.y: test_set_y[index]},
#					mode = self.mode)

		# compute the gradient of cost with respect to theta = (W, W_in, W_out)
		# gradients on the weights using BPTT
		gparams = []
		for param in self.rnn.params:
			gparam = t.grad(cost, param)
			gparams.append(gparam)

#		updates = {}
#		for param, gparam in zip(self.rnn.params, gparams):
#			weight_update = self.rnn.updates[param]
#			upd = weight_update - l_r * gparam
#			updates[weight_update] = upd
#			updates[param] = param + upd

		# compiling a Theano function `train_model` that returns the
		# cost, but in the same time updates the parameter of the
		# model based on the rules defined in `updates`
		train_model = theano.function(inputs = [index, l_r],
				outputs = cost,
#				updates=updates,
				givens={
					self.x: train_set_x[index],
					self.y: train_set_y[index]},
				on_unused_input = "ignore")

		###############
		# TRAIN MODEL #
		###############
#		if self.logger is not None:
#			self.logger.info('... training')
		epoch = 0

		while (epoch < self.n_epochs):
			epoch = epoch + 1
			for idx in xrange(n_train):
#				effective_momentum = self.final_momentum \
#						if epoch > self.momentum_switchover \
#						else self.initial_momentum
				example_cost = train_model(idx, self.learning_rate)

			# iteration number (how many weight updates have we made?)
			# epoch is 1-based, index is 0 based
			iter = (epoch - 1) * n_train + idx + 1

			if iter % validation_frequency == 0:
			# compute loss on training set
				train_losses = [compute_train_error(i)
							for i in xrange(n_train)]
				this_train_loss = np.mean(train_losses)

				if self.interactive:
					test_losses = [compute_test_error(i)
							for i in xrange(n_test)]
					this_test_loss = np.mean(test_losses)

				print('epoch %i, seq %i/%i, train loss %f '
					'lr: %f' % (epoch, idx + 1, n_train,
						this_train_loss, self.learning_rate))

#					if self.logger is not None:
#						self.logger.info('epoch %i, seq %i/%i, tr loss %f '
#							'te loss %f lr: %f' % (epoch, idx + 1, n_train,
#							this_train_loss, this_test_loss,
#							self.learning_rate))
#				else:
#					if self.logger is not None:
#						self.logger.info('epoch %i, seq %i/%i, train loss %f '
#							'lr: %f' % (epoch, idx + 1, n_train,
#							this_train_loss, self.learning_rate))

#			self.learning_rate *= self.learning_rate_decay
'''
