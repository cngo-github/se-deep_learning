import numpy as np
import theano
import theano.tensor as T

class RNN(object):
	"""    Recurrent neural network class
	Supported output types:
	real : linear output units, use mean-squared error
	binary : binary output units, use cross-entropy error
	softmax : single softmax out, use cross-entropy error
	"""
	def __init__(self, input, n_in, n_hidden, n_out, n_cl,
			activation=T.nnet.sigmoid):	
		self.input = input
		self.activation = activation

		# recurrent weights as a shared variable
		W_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
						low=-.01, high=.01),
						dtype=theano.config.floatX)
		self.W = theano.shared(value=W_init, name='W')

		# input to hidden layer weights
		W_in_init = np.asarray(np.random.uniform(size=(n_in, n_hidden),
						low=-.01, high=.01),
						dtype=theano.config.floatX)
		self.W_in = theano.shared(value=W_in_init, name='W_in')

		# hidden to output layer weights (y)
		W_out_init = np.asarray(np.random.uniform(size=(n_hidden, n_out),
						low=-.01, high=.01),
						dtype=theano.config.floatX)
		self.W_out = theano.shared(value=W_out_init, name='W_out')

		# hidden to output layer weights (class)
		W_cl = np.asarray(np.random.uniform(size=(n_hidden, n_cl),
						low=-.01, high=.01),
						dtype=theano.config.floatX)
		self.W_cl = theano.shared(value = W_cl, name = 'W_cl')

		h0_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
		self.h0 = theano.shared(value=h0_init, name='h0')

		self.params = [self.W, self.W_in, self.W_out, self.h0]

	        # for every parameter, we maintain it's last update
        	# the idea here is to use "momentum"
        	# keep moving mostly in the same direction
		self.updates = {}
		for param in self.params:
			init = np.zeros(param.get_value(borrow=True).shape,
					dtype=theano.config.floatX)
			self.updates[param] = theano.shared(init)

		# the hidden state `h` for the entire sequence, and the output for the
		# entire sequence `y` (first dimension is always time)
		[self.h, self.y_pred, self.cl_pred], _ = theano.scan(self.step,
							sequences=self.input,
							outputs_info=[self.h0, None, None])

		# push through softmax, computing vector of class-membership
		# probabilities in symbolic form
		self.y_prob = T.nnet.softmax(self.y_pred)
		self.cl_prob = T.nnet.softmax(self.cl_pred)

		# compute prediction as class whose probability is maximal
		self.y_out = T.argmax(self.y_prob, axis=-1)
		self.cl_prob = T.argmax(self.cl_prob, axis = -1)
		self.loss = lambda y: self.nll_multiclass(y)

	def step(self, x_t, h_tm1):
		h_t = self.activation(T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W))
		y_t = T.dot(h_t, self.W_out)
		cl_t = theano.dot(h_t, self.W_cl)

		return h_t, y_t, cl_t

	def nll_multiclass(self, y):
        	# negative log likelihood based on multiclass cross entropy error
        	# y.shape[0] is (symbolically) the number of rows in y, i.e.,
        	# number of time steps (call it T) in the sequence
        	# T.arange(y.shape[0]) is a symbolic vector which will contain
        	# [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        	# Log-Probabilities (call it LP) with one row per example and
        	# one column per class LP[T.arange(y.shape[0]),y] is a vector
        	# v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        	# LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        	# the mean (across minibatch examples) of the elements in v,
        	# i.e., the mean log-likelihood across the minibatch.
		return -T.mean(T.log(self.y_prob)[T.arange(y.shape[0]), y])
