import theano.tensor as T
import theano
import numpy as np
import reberGrammar

dtype = theano.config.floatX

n_in = 7
n_hid = 10
n_out = 7

v = T.matrix(dtype = dtype)

def sample_weights(sizeX, sizeY):
	values = np.ndarray([sizeX, sizeY], dtype=dtype)

	for dx in xrange(sizeX):
		vals = np.random.uniform(low=-1., high=1.,  size=(sizeY,))
		values[dx,:] = vals

	_,svs,_ = np.linalg.svd(values)
	values = values / svs[0]
	return values

def get_parameter(n_in, n_out, n_hid):
	h0 = theano.shared(np.zeros(n_hid, dtype=dtype))
	W_ih = theano.shared(sample_weights(n_in, n_hid))
	W_hh = theano.shared(sample_weights(n_hid, n_hid))
	W_ho = theano.shared(sample_weights(n_hid, n_out))

	return W_ih, W_hh, W_ho, h0

W_ih, W_hh, W_ho, h0 = get_parameter(n_in, n_out, n_hid)
params = [W_ih, W_hh, W_ho, h0]

def step(x_t, h_tm1, W_ih, W_hh, W_ho):
	h_t = T.nnet.sigmoid(theano.dot(x_t, W_ih) + theano.dot(h_tm1, W_hh))
	y_t = theano.dot(h_t, W_ho)
	y_t = T.nnet.softmax(y_t)

	return [h_t, y_t]

[h_vals, o_vals], _ = theano.scan(fn = step,
									sequences = dict(input=v, taps=[0]),
									outputs_info = [h0, None],
									non_sequences = [W_ih, W_hh, W_ho])

target = T.matrix(dtype = dtype)

lr = np.cast[dtype](1.)
learning_rate = theano.shared(lr)

cost = -T.mean(target * T.log(o_vals) + (1.- target) * T.log(1. - o_vals))

def get_train_functions(cost, v, target):
	gparams = []
	for param in params:
		gparam = T.grad(cost, param)
		gparams.append(gparam)

	updates=[]
	for param, gparam in zip(params, gparams):
		updates.append((param, param - gparam * learning_rate))
	learn_rnn_fn = theano.function(inputs = [v, target],
									outputs = cost,
									updates = updates)

	return learn_rnn_fn

learn_rnn_fn = get_train_functions(cost, v, target)

train_data = reberGrammar.get_n_examples(500)

def train_rnn(train_data, nb_epochs=150):
	train_errors = np.ndarray(nb_epochs)

	for x in range(nb_epochs):
		error = 0.

		for j in range(len(train_data)):
			index = np.random.randint(0, len(train_data))
			i, o = train_data[index]
			train_cost = learn_rnn_fn(i, o)
			error += train_cost
		train_errors[x] = error

	return train_errors

nb_epochs=10
train_errors = train_rnn(train_data, nb_epochs)


