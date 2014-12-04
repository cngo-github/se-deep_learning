from rnn import RNN

import theano.tensor as T
import numpy as np

import theano
#import logging

class Model(object):
	def __init__(self, n_in = 5, n_hidden = 50, n_out = 5,
			n_cl = 5, learning_rate=0.01, n_epochs=100,
			learning_rate_decay=1, activation='sigmoid',
			final_momentum=0.9, initial_momentum=0.5,
			momentum_switchover=5, logger = None,
			mode = theano.Mode(linker='cvm')):
        	self.n_in = int(n_in)
		self.n_hidden = int(n_hidden)
		self.n_out = int(n_out)
		self.n_cl = int(n_cl)
		self.learning_rate = float(learning_rate)
		self.learning_rate_decay = float(learning_rate_decay)
		self.n_epochs = int(n_epochs)
		self.activation = activation
		self.initial_momentum = float(initial_momentum)
		self.final_momentum = float(final_momentum)
		self.momentum_switchover = int(momentum_switchover)
		self.logger = logger
		self.mode = mode

		self.ready()

	def ready(self):
		# input (where first dimension is time)
		self.x = T.matrix()

		# target (where first dimension is time)
		self.y = T.vector(name='y', dtype='int32')

	        # initial hidden state of the RNN
		self.h0 = T.vector()

	        # learning rate
		self.lr = T.scalar()

		if self.activation == 'tanh':
			activation = T.tanh
		elif self.activation == 'sigmoid':
			activation = T.nnet.sigmoid
		elif self.activation == 'relu':
			activation = lambda x: x * (x > 0)
		elif self.activation == 'cappedrelu':
			activation = lambda x: T.minimum(x * (x > 0), 6)
		else:
			raise NotImplementedError

		self.rnn = RNN(input=self.x, n_in=self.n_in, n_hidden=self.n_hidden, 
				n_out=self.n_out, n_cl = self.n_cl,
				activation=activation)

		self.predict_proba = theano.function(inputs=[self.x, ],
				outputs=self.rnn.y_prob, mode = self.mode)
		self.predict = theano.function(inputs=[self.x, ],
				outputs=self.rnn.y_out, mode = self.mode)

	def shared_dataset(self, data_xy):
		""" Load the dataset into shared variables """

		data_x, data_y = data_xy
		shared_x = theano.shared(np.matrix(data_x,
						dtype=theano.config.floatX))

		shared_y = theano.shared(np.asarray(data_y,
						dtype=theano.config.floatX))

		return shared_x, T.cast(shared_y, 'int32')

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

		train_set_x, train_set_y = self.shared_dataset((X_train, Y_train))
		n_train = train_set_x.get_value(borrow=True).shape[0]

		if self.interactive:
			n_test = test_set_x.get_value(borrow=True).shape[0]

		######################
		# BUILD ACTUAL MODEL #
		######################
		if self.logger is not None:
			self.logger.info('... building the model')

		index = T.lscalar('index')    # index to a case
		# learning rate (may change)
		l_r = T.scalar('l_r', dtype=theano.config.floatX)
		mom = T.scalar('mom', dtype=theano.config.floatX)  # momentum

		cost = self.rnn.loss(self.y)

		compute_train_error = theano.function(inputs=[index, ],
						outputs=self.rnn.loss(self.y),
						givens={
							self.x: train_set_x[index],
							self.y: train_set_y[index]},
						mode = self.mode)

		if self.interactive:
			compute_test_error = theano.function(inputs=[index, ],
					outputs=self.rnn.loss(self.y),
					givens={
						self.x: test_set_x[index],
						self.y: test_set_y[index]},
					mode = self.mode)

		# compute the gradient of cost with respect to theta = (W, W_in, W_out)
		# gradients on the weights using BPTT
		gparams = []
		for param in self.rnn.params:
			gparam = T.grad(cost, param)
			gparams.append(gparam)

		updates = {}
		for param, gparam in zip(self.rnn.params, gparams):
			weight_update = self.rnn.updates[param]
			upd = mom * weight_update - l_r * gparam
			updates[weight_update] = upd
			updates[param] = param + upd

		# compiling a Theano function `train_model` that returns the
		# cost, but in the same time updates the parameter of the
		# model based on the rules defined in `updates`
		train_model = theano.function(inputs=[index, l_r, mom],
				outputs=cost,
				updates=updates,
				givens={
					self.x: train_set_x[index],
					self.y: train_set_y[index]},
				mode = self.mode)

		###############
		# TRAIN MODEL #
		###############
		if self.logger is not None:
			self.logger.info('... training')
		epoch = 0

		while (epoch < self.n_epochs):
			epoch = epoch + 1
			for idx in xrange(n_train):
				effective_momentum = self.final_momentum \
						if epoch > self.momentum_switchover \
						else self.initial_momentum
			example_cost = train_model(idx, self.learning_rate,
					effective_momentum)

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

					if self.logger is not None:
						self.logger.info('epoch %i, seq %i/%i, tr loss %f '
							'te loss %f lr: %f' % (epoch, idx + 1, n_train,
							this_train_loss, this_test_loss,
							self.learning_rate))
				else:
					if self.logger is not None:
						self.logger.info('epoch %i, seq %i/%i, train loss %f '
							'lr: %f' % (epoch, idx + 1, n_train,
							this_train_loss, self.learning_rate))

			self.learning_rate *= self.learning_rate_decay
