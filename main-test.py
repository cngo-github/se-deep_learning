import theano
import logging

import theano.tensor as t
import numpy as np

from corpus import Corpus
from model1 import Model

logging.basicConfig(level = logging.INFO)

filepath = 'test'

c = Corpus()
c.buildVocabulary(filepath)
c.saveVocabulary('vocab')

fs = open(filepath, 'r')

#model = None
#x = c.encodeAllTokens(filepath)
#n_in = len(x[0])
#n_hid = 10
#n_out = n_in
#params = {
#	'n_in': n_in,
#	'n_hid': n_hid,
#	'n_out': n_out,
#}
#print(x)
#if not model:
#	model = Model(params)
#y = np.zeros((5, 5), dtype = np.int)
#model.fit([x], x)
#x, target = c.encodeNextLine(fs)
#model.fit(x, y)
#input("waiting")

x, target = c.encodeNextLine(fs)
model = None

while x and target.any():
	n_in = len(x[0])
	n_hid = n_in
	n_out = n_in

	params = {
		'n_in': n_in,
		'n_hid': n_hid,
		'n_out': n_out,
	}

	if not model:
		model = Model(params)

	model.fit(x, target)

	x, target = c.encodeNextLine(fs)

fs.close()

'''
#x = c.encodeAllTokens("test")

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

d = {
	'n_in': n_in,
	'n_hid': n_hid,
	'n_out': n_out,
}

model = Model(d)
model.fit([x], y)
model.save('test.pkl')
model.load('test.pkl')
'''