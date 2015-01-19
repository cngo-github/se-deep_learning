import theano
import logging

import theano.tensor as t
import numpy as np

from corpus import Corpus
from model import Model

logging.basicConfig(level = logging.INFO)

filepath = 'test'

c = Corpus()
c.buildVocabulary(filepath)
c.saveVocabulary('vocab')

fs = open(filepath, 'r')

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