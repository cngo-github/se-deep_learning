'''
	It will parse the corpus file and generate the corpus.  It will also provide the one-hot
	encoding for the specified token.
'''

import queue

class Corpus:
	def __init__(self, in_file, out_file, notify_interval = 100):
		self.vocab = None
		self.encodedTokens = []
		self.in_file = in_file
		self.out_file = out_file
		self.notify_interval = notify_interval

		self.buildVocabulary()

	'''
		Reads the file passed in and generates the vocabulary and list of tokens from it.
	'''
	def buildVocabulary(self):
		print("Reading file: ", self.in_file, end = '')
		inFile = open(self.in_file, 'r')

		unique = set()
		i = 0

		for line in inFile:
			while line:
				pos = line.find(' ')

				if pos < 1:
					pos = len(line)

				token = line[:pos].strip()
				unique.add(token)

				line = line[pos + 1:]
				i += 1
			
			if not i % self.notify_interval:
				i = 0
				print('.', end = '')

		inFile.close()
		self.vocab = list(unique)
		print('\nVocabulary created.')

	'''
		Returns the tokens as a series of one-hot encoded vectors.
	'''
	def encodedAllTokens(self):
		inFile = open(self.in_file, 'r')
		outFile = open(self.out_file, 'w')

		print('Encoding all the tokens in ', self.in_file, ' and writing them to ', \
			self.out_file, end = '')
		i = 0
		for line in inFile:
			while line:
				pos = line.find(' ')

				if pos < 1:
					pos = len(line)

				token = line[:pos].strip()
				line = line[pos + 1:]
				print(self.encodeToken(token), file = outFile)

				i += 1
			
			if not i % self.notify_interval:
				i = 0
				print('.', end = '')

		outFile.close()
		inFile.close()
		print("All the tokens have been encoded.")

	def encodeToken(self, token):
		i = self.vocab.index(token)

		m = [0] * len(self.vocab)
		m[i] = 1

		return m
