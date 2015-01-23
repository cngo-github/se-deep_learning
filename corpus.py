'''
	It will parse the corpus file and generate the corpus.  It will also provide the one-hot
	encoding for the specified token.
'''
import logging

import numpy as np

class Corpus:
	def __init__(self, logger):
		self.vocabulary = np.asarray([])
		self.logger = logger

	'''
		Reads the file passed in and generates the vocabulary and list of tokens from it.
	'''
	def buildVocabulary(self, filepath):
		self.logger.info("Building vocabulary from file: " + filepath)

		fs = open(filepath, 'r')

		for line in fs:
			line = line.strip().split()
			self.vocabulary = np.append(self.vocabulary, line)
			self.vocabulary = np.unique(self.vocabulary)

		fs.close()
		self.logger.info("Vocabulary created.")

	def writeVocabulary(self, infile, outfile):
		'''
			Processes the infile, building a vocabulary and writing it to the outfile.
			This should be used for source corpora that are too large to maintain the
			entire vocabulary of the corpora in memory.
		'''
		self.logger.info("Reading from {} and writing to {}.".format(infile, outfile))
		fs = open(infile, 'r')
		os = open(outfile, 'w')

		for line in fs:
			line = line.strip().split()

			diff = np.setdiff1d(line, self.vocabulary)

			for token in diff:
				os.write(token + '\n')

			self.vocabulary = np.append(self.vocabulary, line)
			self.vocabulary = np.unique(self.vocabulary)
		self.logger.info("Vocabulary from {} and written to {}.".format(infile, outfile))

	def saveVocabulary(self, filepath):
		fs = open(filepath, 'w')

		for s in self.vocabulary:
			fs.write(s + '\n')

		fs.close()

		self.logger.info('Saved vocabulary to file ' + filepath)

	def loadVocabulary(self, filepath):
		fs = open(filepath, 'r')

		self.logger.info("Loading vocabulary from file {}.".format(filepath))

		for line in fs:
			line = line.strip()

			self.vocabulary = np.append(self.vocabulary, line)
			self.vocabulary = np.unique(self.vocabulary)

		self.logger.info("Vocabulary loaded from file {}.".format(filepath))

	def encode(self, sequences, steps, tokens, filestream):
		if tokens is None:
			tokens = np.asarray([])

		min_token_count = steps + sequences + 1

		while len(tokens) < min_token_count:
			# Need to retrieve more tokens to complete the
			# input matrix.
			line = filestream.readline().strip().split()

			if len(line) < 1:
				return (None, None, tokens)

			tokens = np.append(tokens, line)

		targets = None
		inputs = None

		for i in xrange(sequences):
			# Create an input matrix for each sequence.
			mIn = None

			for j in xrange(steps + 1):
				z = self.encodetoken(tokens[i + j])

				if j == (steps):
					# This is the target for the proceeding
					# input.
					if targets == None:
						targets = z
					else:
						targets = np.vstack((targets, z))
				else:
					# Add an encoded token to the input.
					if mIn == None:
						mIn = z
					else:
						mIn = np.vstack((mIn, z))

			# The input 2D arrays are transposed before being stacked to
			# preserve the dimensions.
			if inputs == None:
				inputs = mIn.T
			else:
				inputs = np.dstack((inputs, mIn.T))

		# Transposing inputs prior to returning in order to preserve the dimensions
		# along the expected axises.
		inputs = inputs.T

		a, _ = targets.shape
		targets.resize((a, steps))
		tokens = tokens[-steps:]

		return (inputs, targets, tokens)

	def encodetoken(self, token):
		'''
			One-hot encodes the passed in token.  The length of the encoding
			is equal to the size of the vocabulary.
		'''
		m = np.zeros(len(self.vocabulary))
		i, = (self.vocabulary == token).nonzero()

		if not i:
			self.logger.error('Token not in vocabulary list: %s' % token)
			return m

		m[i[0]] = 1
		return m