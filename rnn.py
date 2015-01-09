import theano
import theano.tensor as t
import numpy as np

class RNN(object):
	def __init__(self, n_in, n_out, n_hid, inputs, learning_rate = 1.,
			dtype = theano.config.floatX, activation = t.nnet.sigmoid,
			output_type = t.nnet.softmax):
		W_ih, W_hh, W_ho, h0 = self.get_parameter(n_in, n_out, n_hid, dtype)
		self.h0 = theano.shared(value = h0, name = 'h0')
		self.W_ih = theano.shared(value = W_ih, name = 'W_ih')
		self.W_hh = theano.shared(value = W_hh, name = 'W_hh')
		self.W_ho = theano.shared(value = W_ho, name = 'W_ho')
		self.params = [W_ih, W_hh, W_ho, h0]

		self.inputs = inputs
		self.activation = activation
		self.output_type = output_type

		[h_vals, o_vals], _ = theano.scan(fn = self.step,
									sequences = dict(input=self.inputs, taps=[0]),
									outputs_info = [self.h0, None],
									non_sequences = [self.W_ih, self.W_hh, self.W_ho])

		self.o_vals = o_vals
		self.h_vals = h_vals

		self.learning_rate = theano.shared(np.cast[dtype](learning_rate))

	def get_parameter(self, n_in, n_out, n_hid, dtype = theano.config.floatX):
		h0 = np.zeros((n_hid,), dtype=dtype)
		W_ih = np.asarray(np.random.uniform(size=(n_in, n_hid),
						low= -.01, high= .01),
						dtype = dtype)
		W_hh = np.asarray(np.random.uniform(size = (n_hid, n_hid),
						low = -.01, high = .01),
						dtype = dtype)
		W_ho = np.asarray(np.random.uniform(size = (n_hid, n_out),
						low = -.01, high = .01),
						dtype = dtype)

		return W_ih, W_hh, W_ho, h0

	def step(self, x_t, h_tm1, W_ih, W_hh, W_ho):
		h_t = self.activation(theano.dot(x_t, W_ih) + theano.dot(h_tm1, W_hh))
		y_t = theano.dot(h_t, W_ho)
		y_t = self.output_type(y_t)

		return [h_t, y_t]

	def loss(self, target):
		return -t.mean(t.log(self.o_vals)[t.arange(target.shape[0]), target])

	def get_train_functions(self, target):
		cost = self.loss(target)
		gparams = []
		params = self.params
		cnt = 0

		index = t.lscalar('index')
#		compute_train_error = theano.function(inputs=[index,],
#											outputs = cost,
#											on_unused_input = 'warn')

		for param in params:
			cnt += 1
			print(cnt)
			gparam = t.grad(cost, param)
			gparams.append(gparam)

		updates=[]
		for param, gparam in zip(params, gparams):
			updates.append((param, param - gparam * self.learning_rate))

		learn_rnn_fn = theano.function(inputs = [index,],
										outputs = cost,
										updates = updates)

		return learn_rnn_fn

	def train_rnn(self, train_data, learn_rnn_fn, nb_epochs = 150):
		train_errors = np.ndarray(nb_epochs)

		for x in range(nb_epochs):
			error = 0.

			for j in range(len(train_data)):
				index = np.random.randint(0, len(train_data))
				i, o = train_data[index]
				train_cost = self.learn_rnn_fn(i, o)
				error += train_cost
			train_errors[x] = error

		return train_errors
