import theano
import theano.tensor as t
#import reberGrammar
import numpy as np
from model import Model

dtype = theano.config.floatX

n_in = 5
n_hid = 10
n_classes = 3
n_out = n_classes

n_seq = 100
n_steps = 10

x = np.random.randn(n_seq, n_steps, n_in)
y = np.zeros((n_seq, n_steps), dtype = np.int)

model = Model(n_in, n_hid, n_out, n_classes)
model.fit(x, y)
