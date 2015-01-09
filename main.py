import theano
import theano.tensor as t
import reberGrammar
from rnn import RNN

dtype = theano.config.floatX

n_in = 7
n_hid = 10
n_out = 7

v = t.matrix(dtype = dtype)
target = t.matrix(dtype = dtype)
target = t.cast(target, 'int32')

rnn = RNN(n_in, n_out, n_hid, v)

learn_rnn_fn = rnn.get_train_functions(target)
train_data = reberGrammar.get_n_examples(500)

nb_epochs=10
train_errors = train_rnn(train_data, learn_rnn_fn, nb_epochs)
