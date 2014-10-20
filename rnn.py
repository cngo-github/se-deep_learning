import numpy as np
import theano
import theano.tensor as T

class RNN:
	def __init__(self, params):
		# recurrent weights as a shared variable
		W = np.asarray(np.random.uniform(size=(params['n_hidden'], params['n_hidden']), low=-.01, high=.01))
		self.W = theano.shared(value = W, name = 'W')
		
		# input to hidden layer weights
		W_in = np.asarray(np.random.uniform(size=(params['n_input'], params['n_hidden']), low=-.01, high=.01))
		self.W_in = theano.shared(value = W_in, name = 'W_in')

		# hidden to output layer weights
		W_out = np.asarray(np.random.uniform(size=(params['n_hidden'], params['n_output']), low=-.01, high=.01))
		self.W_out = theano.shared(value = W_out, name = 'W_out')

		# hidden to class output layer weights
		W_cl = np.asarray(np.random.uniform(size=(params['n_hidden'], params['n_class']), low=-.01, high=.01))
		self.W_cl = theano.shared(value = W_cl, name = 'W_cl')

		# activation function
		self.activation = params['activation']
		self.input = params['input']
		self.h0 = params['h0']

		self.params = [self.W, self.W_in, self.W_out, self.W_cl, self.h0]

		# step function for scan
		def step(self, x_t, h_tm1):
			h_t self.activation(theano.dot(self.W_in, x_t) + theano.dot(self.W, h_tm1))
			y_t = T.softmax(theano.dot(self.W_out, h_t))
			cl_t = T.softmax(theano.dot(self.W_cl, h_t))

		return [h_t, y_t, cl_t]

		[self.h, self.y_pred, self.cl_pred], _ = theano.scan(step, \
						sequences = dict(input = self.input, taps = [-1, 0]) \
						outputs_info = [dict(self.h0, taps = [-1, 0]), None] \
						non_sequences = [self.W, self.W_in, self.W_out, self_cl])

		self.y_out = T.argmax(self.y_pred, self.cl_pred)
		self.loss = lambda y: self.multiclass(y)

	def multiclass(self, y):
		return -T.mean(T.log(self.y_pred)[T.arange(y.shape[0]), y])
