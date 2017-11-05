#coding: utf8
from __future__ import division, print_function

import sys
import codecs
# from langconv import *

embs_50_vocab = {}
vocab_from_conllu_file = {}
embs_100_vocab = {}
embs_300_vocab = {}

# def cht2chs(embs_vocab):
# 	simplified_embs_vocab = {}
# 	duplicates_num = 0

# 	for word, count in embs_vocab.iteritems():
# 		word = Converter('zh-hans').convert(word)
# 		if word in simplified_embs_vocab:
# 			duplicates_num += 1
# 			simplified_embs_vocab[word] += count
# 		else:
# 			simplified_embs_vocab[word] = count

# 	return simplified_embs_vocab, duplicates_num

def save_vocab(embs_vocab, embs_vocab_file):
	with codecs.open(embs_vocab_file, "w", encoding="utf-8") as fo:
		for k, v in embs_vocab.iteritems():
			fo.write(k + "\t\t" + str(v) + "\n")
	print ("{} is saved.".format(embs_vocab_file))

def load_vocab(emb_file):
	embs_vocab = {}

	duplicates_num = 0

	word = ""
	id = 0

	try:
		with codecs.open(emb_file, "r", encoding='utf-8') as fi:
			lines =fi.read().strip().split("\n")
			for line in lines:
				line = line.strip()
				word = line
				id += 1

				if line:
					tokens = line.split()
					if len(tokens) != 2 and tokens[0] != u'</s>':
						# assert len(tokens) == 101 or len(tokens) == 301 or len(tokens) == 51, "dimensions of embedding must be 50, 100 or 300."
						if len(tokens) == 101 or len(tokens) == 301 or len(tokens) == 51:
							word = tokens[0]
							if word not in embs_vocab:
								embs_vocab[word] = 1
							else:
								duplicates_num += 1 if embs_vocab[word] == 1 else 0
								embs_vocab[word] += 1
						else:
							print(len(tokens), line)
	except UnicodeDecodeError as e:
		print (e)
		print (id)
		print (word)
	else:
		pass
	finally:
		pass

	return embs_vocab, duplicates_num

def read_from_conllu_file(conllu_file):
	global vocab_from_conllu_file
	with codecs.open(conllu_file, "r", encoding="utf-8") as fi:
		sentences_list = fi.read().strip().split("\n\n")

		for sentence in sentences_list:
			for line in sentence.strip().split("\n"):
				if line.strip()[0] != "#":
					item = line.strip().split("\t")[1]
					if item in vocab_from_conllu_file:
						vocab_from_conllu_file[item] += 1
					else:
						vocab_from_conllu_file[item] = 1

def compare_vocab(vocab_1, vocab_2):
	total_vocab_1 = len(vocab_1)
	total_vocab_2 = len(vocab_2)
	duplicates_num = 0
	for key in vocab_2.keys():
		duplicates_num += 1 if key in vocab_1 else 0
	print ("Total vocabulary in vocab_1:\t\t\t{}".format(total_vocab_1))
	print ("Total vocabulary in vocab_2:\t\t\t{}".format(total_vocab_2))
	print ("Duplicate vocabulary in the two vocabs:\t{}".format(duplicates_num))


if __name__ == '__main__':
	# word embedding file
	embs_50_file = "/data/ltp/ltp-data/dep/giga-50.bin"
	# embs_100_file = "/data/universal_dependency/embeddings-100/Chinese/zh.vectors"
	# embs_300_file = "/data/universal_dependency/embeddings-300/wiki.zh.vec"

	# obtain vocab
	embs_50_vocab, embs_50_duplicates_num = load_vocab(embs_50_file)
	# embs_100_vocab, embs_100_duplicates_num = load_vocab(embs_100_file)
	# embs_300_vocab, embs_300_duplicates_num = load_vocab(embs_300_file)

	read_from_conllu_file("/users3/dcteng/work/CDT4UD_Chinese/data/ud-simplified/zhs-ud-train.conllu")
	read_from_conllu_file("/users3/dcteng/work/CDT4UD_Chinese/data/ud-simplified/zhs-ud-dev.conllu")
	read_from_conllu_file("/users3/dcteng/work/CDT4UD_Chinese/data/cdt/dep/cdt-train.conll")
	read_from_conllu_file("/users3/dcteng/work/CDT4UD_Chinese/data/cdt/dep/cdt-holdout.conll")
	read_from_conllu_file("/users3/dcteng/work/CDT4UD_Chinese/data/cdt/dep/cdt-test.conll")

	# convert trandional chinese words into simplified chinese words
	# simplified_embs_100_vocab, simplified_embs_100_duplicates_num = cht2chs(embs_100_vocab)
	# simplified_embs_300_vocab, simplified_embs_300_duplicates_num = cht2chs(embs_300_vocab)

	# number of duplicate words
	print  ("Number of Duplicate Words in Embs_50: {}".format(embs_50_duplicates_num))
	# print ("Number of Duplicate Words in Embs_100: {}".format(embs_100_duplicates_num))
	# print ("Number of Duplicate Words in Embs_300: {}".format(embs_300_duplicates_num))
	# print ("Number of Duplicate Words in Simplified_Embs_100: {}".format(simplified_embs_100_duplicates_num))
	# print ("Number of Duplicate Words in Simplified_Embs_300: {}".format(simplified_embs_300_duplicates_num))

	# compare vocab
	compare_vocab(embs_50_vocab, vocab_from_conllu_file)

	# save vocab in embedding file
	embs_50_vocab_file = "resources/embeddings/embs_50_vocab.txt"
	vocab_from_conllu_file_file = "resources/embeddings/vocab_from_conllu_file_file"
	# embs_100_vocab_file = "/users3/dcteng/learn/codes/pytorch/tutorials/multi-task_learning_for_cdt--ud/resources/embeddings/embs_100_vocab.txt"
	# embs_300_vocab_file = "/users3/dcteng/learn/codes/pytorch/tutorials/multi-task_learning_for_cdt--ud/resources/embeddings/embs_300_vocab.txt"
	# simplified_embs_100_vocab_file = "/users3/dcteng/learn/codes/pytorch/tutorials/multi-task_learning_for_cdt--ud/resources/embeddings/simplified_embs_100_vocab.txt"
	# simplified_embs_300_vocab_file = "/users3/dcteng/learn/codes/pytorch/tutorials/multi-task_learning_for_cdt--ud/resources/embeddings/simplified_embs_300_vocab.txt"
	# save_vocab(embs_100_vocab, embs_100_vocab_file)

	save_vocab(embs_50_vocab, embs_50_vocab_file)
	save_vocab(vocab_from_conllu_file, vocab_from_conllu_file_file)
	# save_vocab(embs_300_vocab, embs_300_vocab_file)
	# save_vocab(simplified_embs_100_vocab, simplified_embs_100_vocab_file)
	# save_vocab(simplified_embs_300_vocab, simplified_embs_300_vocab_file)

