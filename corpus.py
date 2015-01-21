'''
	It will parse the corpus file and generate the corpus.  It will also provide the one-hot
	encoding for the specified token.
'''
import logging

import numpy as np

class Corpus:
	def __init__(self):
		self.vocabulary = np.asarray([])

	'''
		Reads the file passed in and generates the vocabulary and list of tokens from it.
	'''
	def buildVocabulary(self, filepath):
		logging.info("Building vocabulary from file: " + filepath)

		fs = open(filepath, 'r')

		for line in fs:
			line = line.strip().split()
			self.vocabulary = np.append(self.vocabulary, line)
			self.vocabulary = np.unique(self.vocabulary)

		fs.close()
		logging.info("Vocabulary created.")

	def writeVocabulary(self, infile, outfile):
		logging.info("Reading from {} and writing to {}.".format(infile, outfile))
		fs = open(infile, 'r')
		os = open(outfile, 'w')

		for line in fs:
			line = line.strip().split()

			diff = np.setdiff1d(line, self.vocabulary)

			for token in diff:
				os.write(token + '\n')

			self.vocabulary = np.append(self.vocabulary, line)
			self.vocabulary = np.unique(self.vocabulary)
		logging.info("Vocabulary from {} and written to {}.".format(infile, outfile))

	def saveVocabulary(self, filepath):
		fs = open(filepath, 'w')

		for s in self.vocabulary:
			fs.write(s + '\n')

		fs.close()

		logging.info('Saved vocabulary to file ' + filepath)

	def loadVocabulary(self, filepath):
		fs = open(filepath, 'r')

		logging.info("Loading vocabulary from file {}.".format(filepath))

		for line in fs:
			line = line.strip()

			self.vocabulary = np.append(self.vocabulary, line)
			self.vocabulary = np.unique(self.vocabulary)

		logging.info("Vocabulary loaded from file {}.".format(filepath))

	def encodeNextLine(self, filestream):
		line = filestream.readline()

		if line == '':
			logging.info('The end of the file has been reached.')
			return (None, None)

		line = line.strip()

		while not line:
			line = filestream.readline()

			if line == '':
				logging.info('The end of the file has been reached.')
				return (None, None)

			line = line.strip()

		if line:
			return self.encodeLine(line)

	def encodeLine(self, line):
		retList = []

		logging.debug('One-hot encoding line: ', line)

		line = line.strip().split()
		target = None

		for token in line:
			target = self.encode(token)
			retList.append(target)

		return (retList, target)

	def encodeLineN(self, line):
		retList = []

		line = line.strip().split()
		target = []

		for token in line:
			a = self.encode(token)
			retList.append(a)

			if len(retList) > 1:
				target.append(a)

		return (retList, target)

	def readNextNotEmptyLine(self, filestream):
		line = fs.readline()

		if line == '':
			logging.info('The end of the file has been reached.')
			return None

		line = line.strip()

		while not line:
			line = filestream.readline()

			if line == '':
				logging.info('The end of the file has been reached.')
				return None

			line = line.strip()

		return line

	def encodeAllTokens(self, inFile):
		inFile = open(inFile, 'r')
		retList = []

		print('Encoding all the tokens in %s.' % inFile)

		for line in inFile:
			lineSplit = line.split()

			for token in lineSplit:
				retList.append(self.encode(token))

		inFile.close()
		print("All the tokens have been encoded.")

		return retList

	def encode(self, sequences, steps, tokens, filestream):
		if not tokens:
			tokens = np.asarray([])

		min_token_count = steps + sequences + 1

		while len(tokens) < min_token_count:
			# Need to retrieve more tokens to complete the
			# input matrix.
			line = filestream.readline().strip().split()
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
		m = np.zeros(len(self.vocabulary))
		i, = (self.vocabulary == token).nonzero()

		if not i:
			logging.error('Token not in vocabulary list: %s' % token)
			return m

		m[i[0]] = 1
		return m