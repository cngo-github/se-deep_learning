import corpus
import rnn
import theano.tensor as T

c = corpus.Corpus('test_corpus', 'test_encoded')
c.getEncodedTokens()
print(c.encodedTokens)

self.rnn = RNN({ 'n_hidden': 5,
		'n_input': 5,
		'n_output': 5,
		'n_class': 5,
		'activation': T.nnet.sigmoid,
		'input': T.matrix(),
		'h0': T.vector()}

shared_x = theano.shared(np.asarray(data_x))
shared_y = theano.shared(np.asarray(data_y))

index = T.lscalar('index')
l_r = T.scalar('l_r')
learning_rate = 0.01
learning_rate_decay = 1 

# compute gradients
gparams = []

for param in self.rnn.params:
	gparams.append(T.grad(self.rnn.loss(y), param))

updates = {}
for param, garam in zip(self.rnn.params, gparams):
	weights_delta = self.rnn.updates[param]
	updates[weights_delta] = weights_delta - l_r * gparam
	updates[param] = param + updates[weights_delta]

train_model = theano.function(inputs = [index, l_r],
			outputs = cost,
			updates = updates,
			givens = {self.x: train_set_x[index],
				self.y: train_set_y[index]})

epoch = 0

while (epoch < self.max_epochs):
	for i in xrange(n_train):
		cost = train_model(i, self.learning_rate)
	epoch += 1

