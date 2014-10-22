import theano
import theano.tensor as T

import rnn

class RNN_Controller:
	def ready(self):
		self.x = T.matrix()
		self.y = T.vector(name = 'y')
		self.h0 = T.vector()
		self.lr = T.scalar()

		if self.activation == 'sigmoid':
			activation = T.nnet.sigmoid
		else:
			raise NotImplementedError

		params = {'input' = self.x,
			'n_input' = self.n_input,
			'n_hidden' = self.n_hidden,
			'n_output' = self.n_output,
			'n_class' = self.n_class,
			'activation' = activation}

		self.rnn = RNN(params)

	def fit(self, x_train, y_train, x_test = None, y_test = None, validation_freq = 100):
		train_set_x = theano.shared(np.asarray(x_train))
		train_set_y = theano.shared(np.asarray(y_train))

		n_train = train_set_x.get_value(borrow = True).shape[0]

		index = T.lscalar('index')
		l_r = T.scalar('l_r')
		cost = self.rnn.loss(self.y)

		compute_train_error = theano.function(inputs = [index, ],
						outputs = self.rnn.lost(self.y),
						givens = {self.x: train_set_x[index],
							self.y: train_set_y[index]})

		gparams = []
		for param in self.rnn.params:
			gparam = T.grad(cost, param)
			gparams.append(gparam)

		updates = {}
		for param, gparam in zip(self.rnn.params, gparams):
			weight_update = self.rnn.updates[param]
			a = weight_update - l_r * gparam
			updates[weight_update] = a
			updates[param] = param + a

		epoch = 0

		while (epoch < self.n_epochs):
			epoch += 1
			for i in xrange(x_train):
				ex_cost = train_model(i, self.learning_rate)

			j = (epoch - 1) * n_train + i + 1
			if j % validation_freq == 0:
				train_losses = [compute_train_error(i) for i in xrange(n_test)]
				train_loss_curr = np.mean(test_losses)
