import numpy as np
import theano
import theano.tensor as TT

nhddn = 50
nin = 5
nout = 5

u = TT.matrix()
t  = TT.matrix()
h0 = TT.vector()
lr = TT.scalar()

# Recurrent weights
w = theano.shared(numpy.random.uniform(size=(n, n), low = -0.01, high = 0.01)

# Input-to-hidden layer weights
w_in = theano.shared(numpy.random.uniform(size=(nin, n), low = -0.01, high = 0.01)

# Hidden-to-output layer weights
w_out = theano.shared(numpy.random.uniform(size=(n, nout), low = -0.01, high = 0.01)

# Hidden-to-output layer weights
w_cl = theano.shared(numpy.random.uniform(size=(n, nout), low = -0.01, high = 0.01)

def step(u_t, z_tm1, w, w_in, w_out, w_cl):
	x_t = TT.nnet.sigmoid(theano.dot(w_in, u_t) + theano.dot(w, z_tm1)
	y_t = TT.nnet.softmax(theano.dot(w_out, x_t)
	cl_t = TT.nnet.softmax(theano.doft(w_cl, x_t)

	return [x_t, y_t, cl_t]

([x_vals, y_vals, cl_vals], updates) = theano.scan(fn = oneStep, \
					sequences = dict(input = u, taps = [-1, -0]), \
					outputs_info = [dict(initial = x0, taps = [-0, -0]), y0] \
					non_sequences = [w, w_in, w_out, w_cl])

# Error between output and target
error = ((y_vals - t) ** 2).sum()

# Gradients on the weights using BPTT
gw, gw_in, gw_out = TT.grad(error, [w, w_in, w_out])

# Training function
fn = theano.function([h0, u, t, lr], error,
			updates = {w: w - lr * gw,
					w_in: w_in - lr * gw_in,
					w_out: w_out - lr * gw_out})
