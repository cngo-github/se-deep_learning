import numpy as np
import theano.tensor as t

import theano

from collections import defaultdict

class RNN(object):
	def __init__(self, input, n_in, n_hid, n_out, activation = t.nnet.sigmoid,
				dtype = theano.config.floatX):
		'''
			input - the shape of the input to be provided.
			n_in - the number of input nodes.
			n_hid - the number of nodes in the hidden layer.
			n_out - the number of nodes in the output layer.
			activation - the activation function to be used by the hidden layer.
						 The default is the sigmoid function.
			dtype - the type to be used for the weights and biases.
		'''
		self.x = input
		self.n_in = n_in
		self.n_hid = n_hid
		self.n_out = n_out
		self.activation = activation
		self.dtype = dtype

		self.ready()

	def ready(self, weights = None):
		'''
			Prepares the RNN for use.
		'''
		# Set the values of the weights
		weights = self.defaultweights(weights)
		self.setweights(weights)

		# Needed for updating.  This defines how the weights are to be updated.
		self.updates = {}

		for param in self.params:
			init = np.zeros(param.get_value(borrow = True).shape,
						dtype = self.dtype)
			self.updates[param] = theano.shared(init)

		# The "RNN" part of the RNN.  h is the hidden state for the sequence
		# and y_pred is the output for the sequence.
		[self.h, self.y_pred], _ = theano.scan(self.step,
										sequences = self.x,
										outputs_info = [self.h0, None])

		# Set the normilizations if they aren't set.
		try:
			self.L1
		except AttributeError:
			self.L1 = 0
			self.L1 += abs(self.W.sum())
			self.L1 += abs(self.W_ih.sum())
			self.L1 += abs(self.W_ho.sum())

		try:
			self.L2_sqr
		except AttributeError:
			self.L2_sqr = 0
			self.L2_sqr += (self.W ** 2).sum()
			self.L2_sqr += (self.W_ih ** 2).sum()
			self.L2_sqr += (self.W_ho ** 2).sum()

		# The softmax signal.
		self.probability_y = t.nnet.softmax(self.y_pred)

		# Calculates the argmax.
		self.y_out = t.argmax(self.probability_y, axis = -1)

	def setweights(self, weights = None):
		'''
			Sets the weights for the RNN based on what is provided.  The weights
			are taken from the weights dictionary that is passed in.  If no
			weights are passed in, then the default values are used.
		'''
		try:
			self.W.set_value(weights.get('W'))
		except AttributeError:
			self.W = theano.shared(value = weights.get('W'), name = 'W')

		try:
			self.W_ih.set_value(weights.get('W_ih'))
		except AttributeError:
			self.W_ih = theano.shared(value = weights.get('W_ih'), name = 'W_ih')

		try:
			self.W_ho.set_value(weights.get('W_ho'))
		except AttributeError:
			self.W_ho = theano.shared(value = weights.get('W_ho'), name = 'W_ho')

		try:
			self.h0.set_value(weights.get('h0'))
		except AttributeError:
			self.h0 = theano.shared(value = weights.get('h0'), name = 'h0')

		self.params = [self.W, self.W_ih, self.W_ho, self.h0]

	def getweights(self):
		d = {
			'W': self.W.get_value(),
			'W_ih': self.W_ih.get_value(),
			'W_ho': self.W_ho.get_value(),
			'h0': self.h0.get_value()
		}

		return d

	def errors(self, y):
		'''
			A float representing the number of errors in the
			sequence.
		'''
		if y.ndim == self.y_out.ndim:
			return t.mean(t.neq(self.y_out, y))

	def step(self, x_t, h_tm1):
		'''
			The calculations that are run every RNN iteration.
		'''
		h_t = self.activation(t.dot(x_t, self.W_ih) + t.dot(h_tm1, self.W))
		y_t = t.dot(h_t, self.W_ho)

		return h_t, y_t

	def loss(self, y):
		'''
			Used to compute the errors between the outputs and the target.
		'''
		return -t.mean(t.log(self.probability_y)[t.arange(y.shape[0]), y])

	def defaultweights(self, weights = None):
		'''
			Provides the default weights for the RNN to use if they are not provided.
			If the weights are provided, the provided weights will be used instead.
		'''
		d = {
			'W': np.asarray(np.random.uniform(size = (self.n_hid, self.n_hid),
							low = -0.01, high = 0.01),
							dtype = self.dtype),
			'W_ih': np.asarray(np.random.uniform(size = (self.n_in, self.n_hid),
							low = -0.01, high = 0.01),
							dtype = self.dtype),
			'W_ho': np.asarray(np.random.uniform(size = (self.n_hid, self.n_out),
							low = -0.01, high = 0.01),
							dtype = self.dtype),
			'h0': np.zeros((self.n_hid,), dtype = self.dtype)
		}

		if weights is None:
			return d

		for key in d.keys():
			weights[key] = weights.get(key) or d.get(keys)

		return weights