from model import Model

import numpy as np
import matplotlib.pyplot as plt

import logging
import time
import theano

logger = logging.getLogger(__name__)
mode = theano.Mode(linker='cvm')
plt.ion()

def test_softmax(n_epochs=250):
    """ Test RNN with softmax outputs. """
    n_hidden = 10
    n_in = 5
    n_steps = 10
    n_seq = 100
    n_classes = 3
    n_out = n_classes  # restricted to single softmax per time step

    np.random.seed(0)
    # simple lag test
    seq = np.random.randn(n_seq, n_steps, n_in)
    print(seq)
    targets = np.zeros((n_seq, n_steps), dtype=np.int)

    thresh = 0.5
    # if lag 1 (dim 3) is greater than lag 2 (dim 0) + thresh
    # class 1
    # if lag 1 (dim 3) is less than lag 2 (dim 0) - thresh
    # class 2
    # if lag 2(dim0) - thresh <= lag 1 (dim 3) <= lag2(dim0) + thresh
    # class 0
    targets[:, 2:][seq[:, 1:-1, 3] > seq[:, :-2, 0] + thresh] = 1
    targets[:, 2:][seq[:, 1:-1, 3] < seq[:, :-2, 0] - thresh] = 2
    #targets[:, 2:, 0] = np.cast[np.int](seq[:, 1:-1, 3] > seq[:, :-2, 0])

    model = Model(logger = logger, mode = mode, n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    learning_rate=0.001, learning_rate_decay=0.999,
                    n_epochs=n_epochs, activation='sigmoid')

    model.fit(seq, targets, validation_frequency=1000)

    seqs = xrange(10)

    plt.close('all')

    for seq_num in seqs:
        fig = plt.figure()
        ax1 = plt.subplot(211)
        plt.plot(seq[seq_num])
        ax1.set_title('input')
        ax2 = plt.subplot(212)

        # blue line will represent true classes
        true_targets = plt.step(xrange(n_steps), targets[seq_num], marker='o')

        # show probabilities (in b/w) output by model
        guess = model.predict_proba(seq[seq_num])
        guessed_probs = plt.imshow(guess.T, interpolation='nearest',
                                   cmap='gray')
        ax2.set_title('blue: true class, grayscale: probs assigned by model')

if __name__ == "__main__":
	logging.basicConfig(filename = 'log', level=logging.INFO)
	print "Starting run"
	t0 = time.time()
	test_softmax(n_epochs=250)
	logger.info( "Elapsed time: %f" % (time.time() - t0))
	print "Elapsed time: %f" % (time.time() - t0)
