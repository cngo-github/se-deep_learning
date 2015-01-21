import numpy as np
import theano.tensor as t

from rnn import RNN

import theano
import logging
import time

class Model(object):
	def __init__(self, logger, params = None):
		self.logger = logger

		self.ready(params)

	def ready(self, params = None):
		'''
			Sets up the model.
		'''
		self.x = t.matrix()
		self.y = t.vector(name = 'y', dtype = 'int32')
		self.h0 = t.vector()
		self.lr = t.scalar()

		params = self.defaultparams(params)
		self.setparams(params)

		self.rnn = RNN(input = self.x, n_in = self.n_in, n_hid = self.n_hid,
						n_out = self.n_out, activation = self.activation)

		self.predict_probability = theano.function(inputs = [self.x,],
												outputs = self.rnn.probability_y)
		self.predict = theano.function(inputs = [self.x,],
										outputs = self.rnn.y_out)

	def fit(self, x_train, y_train, x_test = None, y_test = None, validation_freq = 100):
		'''
			Creates and trains the model.
		'''
		if x_test is not None and y_test is not None:
			self.runtests = True
			test_x, test_y = self.share_dataset(x_test, y_test)
		else:
			self.runtests = False

		train_x, train_y = self.share_dataset(x_train, y_train)
		n_train = train_x.get_value(borrow = True).shape[0]

		'''
			Creates the model.
		'''
		self.logger.info('Building the model...')

		idx = t.lscalar('index')
		l_r = t.scalar(name = 'l_r', dtype = theano.config.floatX)
		mom = t.scalar(name = 'mom', dtype = theano.config.floatX)

		cost = self.rnn.loss(self.y) + self.L1_reg * self.rnn.L1 \
				+ self.L2_reg * self.rnn.L2_sqr

		train_error = theano.function(inputs = [idx,],
									outputs = self.rnn.loss(self.y),
									givens = {
										self.x: train_x[idx],
										self.y: train_y[idx]
									})

		if self.runtests:
			test_error = theano.function(inputs = [idx,],
									outputs = self.rnn.loss(self.y),
									givens = {
										self.x: test_x[idx],
										self.y: test_y[idx]
									})

		# Compute the cost gradients with BPTT
		gparams = []
		for param in self.rnn.params:
			gparam = t.grad(cost, param)
			gparams.append(gparam)

		updates = {}
		for param, gparam in zip(self.rnn.params, gparams):
			update = self.rnn.updates[param]
			u = mom * update - l_r * gparam

			updates[update] = u
			updates[param] = param + u

		# The function to train the model.
		train_model = theano.function(inputs = [idx, l_r, mom],
									outputs = cost,
									updates = updates,
									givens = {
										self.x: train_x[idx],
										self.y: train_y[idx]
									})

		'''
			Train the model
		'''
		self.logger.info('Training the model...')
		epoch = 0

		while epoch < self.n_epochs:
			epoch += 1

			for i in xrange(n_train):
				t0 = time.time()

				eff_momentum = self.final_momentum \
									if epoch > self.momentum_switchover \
									else self.initial_momentum
				example_cost = train_model(i, self.learning_rate, eff_momentum)

				itr = (epoch - 1) * n_train + i + 1

				if itr % validation_freq == 0:
					train_losses = [train_error(j) for j in xrange(n_train)]
					train_losses = np.mean(train_losses)

					if self.runtests:
						test_losses = [test_error(j) for j in xrange(n_test)]
						test_losses = np.mean(test_losses)

						self.logger.info('epoch {}, seq {} / {}, training losses {}, test losses {}, learning rate {}, elasped time {}.'.format(
											epoch, i + 1, n_train, train_losses,
											test_losses, self.learning_rate, time.time() - t0))
					else:
						self.logger.info('epoch {}, seq {} / {}, training losses {}, learning rate {}, elasped time {}.'.format(
											epoch, i + 1, n_train, train_losses,
											self.learning_rate, time.time() - t0))


	def share_dataset(self, data_x, data_y):
		'''
			Load the datasets into shared variables.
		'''
		shared_x = theano.shared(np.asarray(data_x, dtype = theano.config.floatX))
		shared_y = theano.shared(np.asarray(data_y, dtype = theano.config.floatX))

		return shared_x, t.cast(shared_y, 'int32')

	def __getstate__(self):
		'''
			Returns the current state of the model and RNN.
		'''
		params = self.getparams()
		weights = self.rnn.getweights()

		return (params, weights)

	def __setstate__(self, state):
		'''
			Sets the parameters for the model and RNN.
		'''
		params, weights = state

		self.setparams(params)
		self.ready()
		self.rnn.setweights(weights)

	def load(self, path):
		'''
			Unpickles a pickled model.
		'''
		fs = open(path, 'rb')

		self.logger.info('Model state loading from file {}.'.format(path))

		state = pickle.load(fs)
		self.__setstate__(state)

		fs.close()

		self.logger.info('Model state loaded.')

	def save(self, path = None):
		'''
			Pickles the model.
		'''
		if path is None:
			path = str(uuid.uuid4())

		fs = open(path, 'wb')

		state = self.__getstate__()
		pickle.dump(state, fs, protocol = pickle.HIGHEST_PROTOCOL)

		fs.close()

		self.logger.info('Model state saved to file {}.'.format(path))

	def setparams(self, params):
		'''
			Sets the parameters of the model and RNN.
		'''
		self.n_in = params.get('n_in')
		self.n_hid = params.get('n_hid')
		self.n_out = params.get('n_out')
		self.n_epochs = params.get('n_epochs')
		self.learning_rate = params.get('learning_rate')
		self.activation = params.get('activation')
		self.L1_reg = params.get('L1_reg')
		self.L2_reg = params.get('L2_reg')
		self.initial_momentum = params.get('initial_momentum')
		self.final_momentum = params.get('final_momentum')
		self.momentum_switchover = params.get('momentum_switchover')

	def getparams(self):
		'''
			Gets the parameters of the model.
		'''
		d = {
			'n_in': self.n_in,
			'n_hid': self.n_hid,
			'n_out': self.n_out,
			'n_epochs': self.n_epochs,
			'learning_rate': self.learning_rate,
			'activation': self.activation,
			'L1_reg': self.L1_reg,
			'L2_reg': self.L2_reg,
			'initial_momentum': self.initial_momentum,
			'final_momentum': self.final_momentum,
			'momentum_switchover': self.momentum_switchover
		}

		return d

	def defaultparams(self, params = None):
		'''
			Returns the default parameters for the model or
			ensures that all the necessary parameters are
			present.
		'''
		d = {
			'n_in': 5,
			'n_hid': 50,
			'n_out': 5,
			'n_epochs': 100,
			'learning_rate': 0.01,
			'activation': t.nnet.sigmoid,
			'L1_reg': 0.0,
			'L2_reg': 0.0,
			'initial_momentum': 0.5,
			'final_momentum': 0.9,
			'momentum_switchover': 5
		}

		if params is None:
			return d

		for key in d.keys():
			params[key] = params.get(key) or d.get(key)

		return params