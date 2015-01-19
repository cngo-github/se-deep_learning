import theano
import theano.tensor as t
import numpy as np

class RNN(object):
	def __init__(self, params):
		self.setparams(params)
		self.setscanfn()
		self.output()

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

		if not params.has_key('input'):
			raise ValueError("Missing required parameter: input")
		else:
			self.input = params['input']

		self.setdefaultweights(n_in, n_hid, n_out, dtype)

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

	def setscanfn(self):
		[self.h, self.y_pred], _ = theano.scan(fn = self.step,
									sequences = dict(input=self.input),
									outputs_info = [self.h0, None])

	def setdefaultweights(self, n_in, n_out, n_hid, dtype = theano.config.floatX):
		h0 = np.zeros((n_hid,), dtype=dtype)
		self.h0 = theano.shared(value = h0, name = 'h0')

		W_ih = np.asarray(np.random.uniform(size=(n_in, n_hid),
						low= -.01, high= .01),
						dtype = dtype)
		self.W_ih = theano.shared(value = W_ih, name = 'W_ih')

		W_hh = np.asarray(np.random.uniform(size = (n_hid, n_hid),
						low = -.01, high = .01),
						dtype = dtype)
		self.W_hh = theano.shared(value = W_hh, name = 'W_hh')

		W_ho = np.asarray(np.random.uniform(size = (n_hid, n_out),
						low = -.01, high = .01),
						dtype = dtype)
		self.W_ho = theano.shared(value = W_ho, name = 'W_ho')

		self.params = [self.W_ih, self.W_hh, self.W_ho, self.h0]

	def step(self, x_t, h_tm1):
		h_t = self.activation(theano.dot(x_t, self.W_ih) + theano.dot(h_tm1, self.W_hh))
		y_t = theano.dot(h_t, self.W_ho)

		return [h_t, y_t]

	def output(self):
		self.prob_y = self.activation(self.y_pred)
		self.y_out = t.argmax(self.prob_y, axis = -1)

	def loss(self, target):
		return -t.mean(t.log(self.prob_y)[t.arange(target.shape[0]), target])