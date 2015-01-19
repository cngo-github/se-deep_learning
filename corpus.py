'''
	It will parse the corpus file and generate the corpus.  It will also provide the one-hot
	encoding for the specified token.
'''
import logging

import numpy as np

class Corpus:
	def __init__(self):
		self.vocabulary = []

	'''
		Reads the file passed in and generates the vocabulary and list of tokens from it.
	'''
	def buildVocabulary(self, filepath):
		logging.info("Building vocabulary from file: " + filepath)

		fs = open(filepath, 'r')
		tmp = set()

		for line in fs:
			line = line.strip().split()
			tmp.update(line)

		fs.close()
		self.vocabulary = list(tmp)
		logging.info("Vocabulary created.")

	def saveVocabulary(self, filepath):
		fs = open(filepath, 'w')

		for s in self.vocabulary:
			fs.write(s + '\n')

		fs.close()

		logging.info('Saved vocabulary to file ' + filepath)

	def loadVocabulary(self, filepath):
		fs = open(filepath, 'r')
		tmp = set()

		for line in fs:
			tmp.update(line.strip())

		self.vocabulary = list(tmp)
		logging.info('Vocabulary loaded from file ', filepath)

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
			print(line)
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

	def encode(self, token):
		m = np.zeros(len(self.vocabulary))

		try:
			i = self.vocabulary.index(token)
			m[i] = 1
		except ValueError:
			logging.error('Token not in vocabulary list: ', token)

		return m
