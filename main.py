import corpus
import rnn
import theano.tensor as T

import model

c = corpus.Corpus('training_corpus', 'test_encoded')

print(c.input_size)
#for lst in c.classes:
#	if not lst:
#		print("empty at index: ", i)
#print(c.encodedTokens)
'''
n_input, n_hidden, n_output, n_class, learning_rate = 0.01, n_epochs = 100, activation = 'sigmoid'

aModel = Model({n_input: 5,
		n_hidden: 10,
		n_output: 10,
		n_class: 10})

aModel.fit(seq, targets)
'''
