'''
	It will parse the corpus file and generate the corpus.  It will also provide the one-hot
	encoding for the specified token.
'''

import queue
import math

class Corpus:
	def __init__(self, in_file, out_file, total_classes = None, notify_interval = 100, debug_mode = None):
		self.vocab = None
		self.encodedTokens = []
		self.in_file = in_file
		self.out_file = out_file
		self.notify_interval = notify_interval
		self.total_classes = total_classes
		self.debug_mode = debug_mode

		self.buildVocabulary()
		self.calcTokenProbability()
		self.buildClasses()

	'''
		Reads the file passed in and generates the vocabulary and list of tokens from it.
	'''
	def buildVocabulary(self):
		print("Reading file: ", self.in_file, end = '')
		inFile = open(self.in_file, 'r')

		self.vocabulary = {}
		i = 0

		for line in inFile:
			line = line.strip()

			while line:
				pos = line.find(' ')

				if pos < 1:
					pos = len(line)

				token = line[:pos].strip()

				if token in self.vocabulary:
					self.vocabulary[token]['count'] += 1
				else:
					self.vocabulary[token] = {'count': 1}

				line = line[pos + 1:].strip()
				i += 1

		inFile.close()
		self.vocab_token_count = i
		self.input_size = len(self.vocabulary)

		print('\nVocabulary created.  Calculating classes.')

	def calcTokenProbability(self):
		max_probability = 0
		min_probability = 100

		for k, v in self.vocabulary.items():
			prob = (v['count'] / self.vocab_token_count) * 100
#			print('token: ', k, ' count: ', v['count'], ' prob: ', prob)
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

#			print('token: ', k, ' prob: ', v['probability'], ' i: ', i, ' j: ', j, ' l: ', l, ' m: ', m)
			self.classes[prob].append(k)

#		print('max: ', self.max_probability, ' min: ', self.min_probability, ' class: ', class_size)
#		print('classes: ', self.classes[3])

		unfilled = True;
		while unfilled:
			empty = []
			j = 0

			for i in range(class_cnt):
				if not self.classes[i]:
#					print('Empty at: ', i)
					empty.append(i)
					j += 1
				elif empty:
					self.smoothClasses(empty, i)
					empty = []

			if not empty and j > 0:
				unfilled = True
			else:
				unfilled = False

#		for i in range(class_cnt):
#			if not self.classes[i]:
#				print('Empty at: ', i)
#			else:
#				print('Filled at: ', i)
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

		m = [0] * self.input_size
		m[i] = 1

		return m
