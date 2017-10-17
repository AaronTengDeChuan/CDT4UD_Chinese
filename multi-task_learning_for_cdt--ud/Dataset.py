from __future__ import division
import os
import sys
import codecs
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ConlluPOSDataset(Dataset):
	"""Conllu part-of-speech dataset."""

	def __init__(self, data, root_dir):
		super(ConlluPOSDataset, self).__init__()
		self.root_dir = root_dir
		self.data = data
		self.data_size = len(self.data)

		self.padding()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

	def padding(self):
		max_len = 0
		for dict in self.data:
			max_len = dict["length"] if dict["length"] > max_len else max_len
		for id in range(self.__len__()):
			# self.data[id]["sentence"] = \
				self.data[id]["sentence"].extend([0] * (max_len - self.data[id]["length"]))
				self.data[id]["sentence"] = torch.LongTensor(self.data[id]["sentence"])
			# self.data[id]["sentence"].dtype = "int64"
			# self.data[id]["pos_tags"] = \
				self.data[id]["pos_tags"].extend([100] * (max_len - self.data[id]["length"]))
				self.data[id]["pos_tags"] = torch.LongTensor(self.data[id]["pos_tags"])
			# self.data[id]["pos_tags"].dtype = "int64"

class Map():
	"""Map words and tags to id."""
	def __init__(self):
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "PAD"}
		self.num_words = 1

		self.cdt_pos_tag2index = {}
		self.index2cdt_pos_tag = {	0: 'p', 1: 'nt', 2: 'a', 3: 'wp', 4: 'r', 5: 'd', \
									6: 'n', 7: 'nl', 8: 'v', 9: 'i', 10: 'j', 11: 'nd', \
									12: 'u', 13: 'q', 14: 'ns', 15: 'nz', 16: 'm', 17: 'c', \
									18: 'nh', 19: 'b', 20: 'ni', 21: 'k', 22: 'h', 23: 'e', \
									24: 'o', 25: 'ws', 26: 'g', 27: 'x', 28: 'z', 100: 'PAD'}
		self.num_cdt_pos_tags = 29

		self.ud_pos_tag2index = {}
		self.index2ud_pos_tag = {	0: 'INTJ', 1: 'AUX', 2: 'ADJ', 3: 'PUNCT', 4: 'ADV', 5: 'VERB', \
								 	6: 'NUM', 7: 'NOUN', 8: 'PRON', 9: 'PART', 10: 'X', 11: 'ADP', \
									12: 'DET', 13: 'CCONJ', 14: 'PROPN', 15: 'SYM', 16: 'SCONJ', 100: 'PAD'}
		self.num_ud_pos_tags = 17

		self.addTag()

	def read_from_conllu_file(self, conllu_file, UD_Data=True):
		with codecs.open(conllu_file, "r", encoding="utf-8") as fi:
			sentences_list = fi.read().strip().split("\n\n")

			self.sentences = \
			[
				[
					[
						item for id, item in enumerate(line.strip().split("\t"))
							if id == 1 or id == 3
					]
					for line in sentence.strip().split("\n")
						if line.strip()[0] != "#"
				]
				for sentence in sentences_list
			]

		return self.addSentences(self.sentences, UD_Data)

	def addSentences(self, sentences, UD_Data):
		return 	[
					{
						"sentence": [self.addWord(word[0])
							for word in sentence],
						"pos_tags": [self.ud_pos_tag2index[word[1]] if UD_Data else self.cdt_pos_tag2index[word[1]]
							for word in sentence],
						"length": len(sentence)
					}
					for sentence in sentences
				]

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.num_words
			self.word2count[word] = 1
			self.index2word[self.num_words] = word
			self.num_words += 1
		else:
			self.word2count[word] += 1

		return self.word2index[word]

	def addTag(self):
		self.ud_pos_tag2index = dict((v, k) for k, v in self.index2ud_pos_tag.iteritems())
		self.cdt_pos_tag2index = dict((v, k) for k, v in self.index2cdt_pos_tag.iteritems())

	def add_Tag(self, tag):
		if tag not in self.cdt_pos_tag2index:
			self.cdt_pos_tag2index[tag] = self.num_cdt_pos_tags
			self.index2cdt_pos_tag[self.num_cdt_pos_tags] = tag
			self.num_cdt_pos_tags += 1
		return self.cdt_pos_tag2index[tag]

 

if __name__ == "__main__":
	# assert len(sys.argv) == 2, "Usage: python Dataset.py conllu_file"
	# ud_dataset = ConlluPOSDataset("/users3/dcteng/work/cdt-ud/data/ud-traditional/UD_Chinese/zh-ud-dev.conllu", \
	#	"/users3/dcteng/work/cdt-ud/data/ud-traditional/UD_Chinese/")
	mapping = Map()
	data = mapping.read_from_conllu_file("/users3/dcteng/work/cdt-ud/data/ud-traditional/UD_Chinese/zh-ud-train.conllu")
	print "Number of Sentences:\t{}\nNumber of Words:\t{}\nNumber of CDT_Tags:\t{}\nNumber of UD_Tags:\t{}\n"\
		.format(len(data), mapping.num_words, mapping.num_cdt_pos_tags, mapping.num_ud_pos_tags)
	print "UD_Tags:\n{}\n".format(mapping.ud_pos_tag2index)
	UD_Dataset = ConlluPOSDataset(data, "/users3/dcteng/work/cdt-ud/data/ud-traditional/UD_Chinese/")
	UD_Dataloader = DataLoader(UD_Dataset, batch_size=4, shuffle=True)

	for i_batch, sample_batched in enumerate(UD_Dataloader):
		# mini_batch = torch.cat(sample_batched['sentence'], dim=0)
		# mini_batch = torch.zeros(len(sample_batched["sentence"]),4)
		# [mini_batch[i].add_(sample_batched["sentence"][i]) for i in range(len(sample_batched["sentence"]))]
		# mini_batch = torch.stack(sample_batched["sentence"])
		# print i_batch, mini_batch.size()
		# print sample_batched["pos_tags"].size()
		# print sample_batched["length"].size()
		# print len(sample_batched['sentence']), type(sample_batched['sentence']), type(sample_batched['sentence'][0]), sample_batched['sentence'][0].size()
		# print len(sample_batched['pos_tags']), type(sample_batched['pos_tags']), type(sample_batched['pos_tags'][0]), sample_batched['pos_tags'][0].size()
		# print type(sample_batched['length']), sample_batched['length'].size()

		if i_batch == 3:
			# print mini_batch
			for i in range(sample_batched["length"][0]):
				print mapping.index2word[sample_batched["sentence"][0][i]],
			print
			for i in range(sample_batched["length"][0]):
				print mapping.index2ud_pos_tag[sample_batched["pos_tags"][0][i]],
			# print sample_batched['sentence'][sample_batched['length'][0]-1]
			# print sample_batched['pos_tags'][sample_batched['length'][0]]
			# print sample_batched['length']
			break
