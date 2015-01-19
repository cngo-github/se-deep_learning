import theano
import theano.tensor as t
import numpy as np

class RNN(object):
	def __init__(self, params):
		self.setparams(params)
		self.ready()

	def ready(self):
		self.x = t.matrix(name = 'x')
		self.y = t.ivector(name = 'y')
		self.h0 = t.matrix()
		self.lr = t.fscalar()

		h_t, updates = theano.scan(fn = self.step,
									sequences = dict(input= self.x, taps = [0]),
									outputs_info = dict(initial = self.h0, taps = self.output_taps),
									non_sequences = self.params)

		self.act_y = self.output_type(t.dot(h_t, self.W_ho))
		self.y_out = t.argmax(self.act_y, axis = -1)
		self.cost = -t.mean(t.log(self.act_y)[t.arange(self.y.shape[0]), self.y])

	def setparams(self, params):
		if not params.has_key('activation'):
			self.activation = t.nnet.sigmoid
		else:
			self.activation = params['activation']

		if not params.has_key('output_type'):
			self.output_type = t.nnet.softmax
		else:
			self.output_type = params['output_type']

		if not params.has_key('dtype'):
			dtype = theano.config.floatX
		else:
			dtype = params['dtype']

		if not params.has_key('output_taps'):
			self.output_taps = [-1, 0]
		else:
			self.output_taps = params['output_taps']

		#Check for required parameters: n_in, n_hid, n_out, input
		if not params.has_key('n_in'):
			raise ValueError("Missing required parameter: n_in")
		else:
			n_in = params['n_in']

		if not params.has_key('n_hid'):
			raise ValueError("Missing required parameter: n_hid")
		else:
			n_hid = params['n_hid']

		if not params.has_key('n_out'):
			raise ValueError("Missing required parameter: n_out")
		else:
			n_out = params['n_out']

		self.setdefaultweights(n_in, n_hid, n_out, dtype)

	def gettrainfn(self, train_x, train_y, learning_rate):
		in_idx0 = t.iscalar(name = 'input_start')
		in_idx1 = t.iscalar(name = 'input_stop')
		tgt_idx0 = t.iscalar(name = 'target_start')
		tgt_idx1 = t.iscalar(name = 'target_stop')

		gparams = []

		for param in self.params:
			gparam = t.grad(self.cost, param, disconnected_inputs = 'warn')
			gparams.append(gparam)

		updates = {}
		for param, gparam in zip(self.params, gparams):
			updates[param] = param - self.lr * gparam

		trainfn = theano.function(inputs = [in_idx0] + [in_idx1] + [tgt_idx0] + [tgt_idx1],
									outputs = self.cost,
									updates = updates,
									givens = {
										self.x: train_x[in_idx0:in_idx1],
										self.y: train_y[tgt_idx0:tgt_idx1],
										self.h0: t.cast(self.h, 'float64'),
										self.lr: t.cast(learning_rate, 'float32')
									})

		return trainfn

	def getweights(self):
		weights = [w.get_value() for w in self.params]

		weights = {
			'W_ih': self.W_ih,
			'W_hh': self.W_hh,
			'W_ho': self.W_ho,
			'h0': self.h0
		}

		return weights

	def setweights(self, weights):
		self.W_ih = weights['W_ih']
		self.W_hh = weights['W_hh']
		self.W_ho = weights['W_ho']
		self.h0 = weights['h0']

		self.params = [self.W_ih, self.W_hh, self.W_ho, self.h0]

#	def setscanfn(self):
#		self.h_t, updates = theano.scan(fn = self.step,
#									sequences = dict(input=self.input, taps = [0]),
#									outputs_info = dict(initial = self.h0, taps = self.output_taps),
#									non_sequences = self.params)

	def setdefaultweights(self, n_in, n_out, n_hid, dtype = theano.config.floatX):
		# Recurrent activations
		h =np.zeros((n_hid, n_hid), dtype = dtype)
		self.h = theano.shared(value = h, name = 'h')

		# Input to hidden layer weights
		W_ih = np.asarray(np.random.uniform(size=(n_in, n_hid),
						low= -.01, high= .01),
						dtype = dtype)
		self.W_ih = theano.shared(value = W_ih, name = 'W_ih')

		# Recurrent weights
		W_hh = np.asarray(np.random.uniform(size = (n_hid, n_hid),
						low = -.01, high = .01),
						dtype = dtype)
		self.W_hh = theano.shared(value = W_hh, name = 'W_hh')

		# hidden to output layer weights
		W_ho = np.asarray(np.random.uniform(size = (n_hid, n_out),
						low = -.01, high = .01),
						dtype = dtype)
		self.W_ho = theano.shared(value = W_ho, name = 'W_ho')

		self.params = [self.W_hh, self.W_ih, self.W_ho]

	def step(self, x_t, *args):
		act_recurrent = [args[x] for x in xrange(len(self.output_taps))]
		weights_recurrent = args[len(self.output_taps)]

		W_ih = args[len(self.output_taps) * 2]

		activations = theano.dot(act_recurrent[0], weights_recurrent[0])

		for x in xrange(1, len(self.output_taps)):
			activations += t.dot(act_recurrent[x], weights_recurrent[x])

		h_t = self.activation(theano.dot(x_t, W_ih) + activations)

		return h_t