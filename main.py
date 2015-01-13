import theano
import theano.tensor as t
import numpy as np

from corpus import Corpus
from model import Model

c = Corpus("test")

x = c.encodeAllTokens("test")

if len(x):
	n_in = len(x[0])


dtype = theano.config.floatX

#n_in = 5
n_hid = 10
n_classes = n_in
n_out = n_classes

n_seq = 100
n_steps = 10

#x = np.random.randn(n_seq, n_steps, n_in)
y = np.zeros((n_seq, n_steps), dtype = np.int)

#x = np.asanyarray(x, dtype = np.float)
#x.astype(np.float)

model = Model(n_in, n_hid, n_out, n_classes)
model.fit([x], y)
