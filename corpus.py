'''
	It will parse the corpus file and generate the corpus.  It will also provide the one-hot
	encoding for the specified token.
'''

import math

class Corpus:
	def __init__(self, in_file, total_classes = None):
		self.in_file = in_file
		self.total_classes = total_classes

		self.buildVocabulary()
#		self.calcTokenProbability()
#		self.buildClasses()

	'''
		Reads the file passed in and generates the vocabulary and list of tokens from it.
	'''
	def buildVocabulary(self):
		print("Reading file: ", self.in_file)
		inFile = open(self.in_file, 'r')

		self.vocabulary = {}

		for line in inFile:
			line = line.strip()
			lineSplit = line.split()

			for token in lineSplit:
				if token in self.vocabulary:
					self.vocabulary[token]['count'] += 1
				else:
					self.vocabulary[token] = {'count': 1}

		inFile.close()
		self.vocabList = list(self.vocabulary)
		print('\nVocabulary created.')

	def calcTokenProbability(self):
		max_probability = 0
		min_probability = 100

		for k, v in self.vocabulary.items():
			prob = (v['count'] / self.vocab_token_count) * 100
			self.vocabulary[k]['probability'] = prob

			if max_probability < prob:
				max_probability = prob

			if min_probability > prob:
				min_probability = prob

		self.max_probability = max_probability
		self.min_probability = min_probability

	def buildClasses(self):
		if not self.total_classes:
			class_cnt = math.ceil(math.sqrt(self.input_size))
		else:
			class_cnt = self.total_classes

		class_size = (self.max_probability - self.min_probability) / class_cnt

		self.classes = []
		for i in range(class_cnt):
			self.classes.append([])

		for k, v in self.vocabulary.items():
			prob = (self.max_probability - v['probability']) / class_size
			prob = int(prob)

			if prob >= class_cnt:
				prob = class_cnt - 1

			self.classes[prob].append(k)

		unfilled = True;
		while unfilled:
			empty = []
			j = 0

			for i in range(class_cnt):
				if not self.classes[i]:
					empty.append(i)
					j += 1
				elif empty:
					self.smoothClasses(empty, i)
					empty = []

			if not empty and j > 0:
				unfilled = True
			else:
				unfilled = False

		self.max_class_size = 0
		for i in self.classes:
			if self.max_class_size < len(i):
				self.max_class_size = len(i)

	def smoothClasses(self, empty_list, populated_index):
		source = self.classes[populated_index]
		self.classes[populated_index] = []

		source = sorted(source, key = lambda token: self.vocabulary[token]['count'])

		empty_list.append(populated_index)

		distrib_cnt = int(len(source) / len(empty_list))

		if distrib_cnt < 1:
			distrib_cnt = 1

		while empty_list:
			i = empty_list.pop(0)
			j = 0

			while j < distrib_cnt and source:
				self.classes[i].append(source.pop())
				j += 1

		while source:
			self.classes[populated_index].append(source.pop())

	'''
		Returns the tokens as a series of one-hot encoded vectors.
	'''
	def encodeAllTokens(self, inFile):
		inFile = open(inFile, 'r')
		retList = []

		print('Encoding all the tokens in %s.' % inFile)

		for line in inFile:
			lineSplit = line.split()

			for token in lineSplit:
				print(token)
				retList.append(self.encodeToken(token))

		inFile.close()
		print("All the tokens have been encoded.")

		return retList

	def encodeToken(self, token):
		m = [0] * self.getVocabSize()

		try:
			i = self.vocabList.index(token)
			m[i] = 1
		except ValueError:
			print('Token ' + token + ' not in vocabulary list.')

		return m
