from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random

from Dataset import ConlluPOSDataset, Map
from torch.utils.data import Dataset, DataLoader

from Bi_LSTM_Model import MultiTaskBiLSTM

import time
import math
import codecs
from collections import Iterable, Iterator

def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)

# def inputTensor(self, input):
# 	inputs = torch.LongTensor(len(input), len(input[0]))
# 	for i in range(len(input)):
# 		inputs[i] = input[i]
# 	return inputs

# def targetTensor(self, target):
# 	targets = torch.LongTensor(len(target), len(target[0]))
# 	for i in range(len(target)):
# 		targets[i] = target[i]
# 	return targets

def trainExamples(sample_batched):
	max_len = sample_batched["length"].max()
	inputs = Variable(sample_batched["sentence"][:, :max_len].contiguous().cuda())
	targets = Variable(sample_batched["pos_tags"][:, :max_len].contiguous().cuda())
	lengths = Variable(sample_batched["length"].cuda())

	return inputs, targets, lengths

def output_result(result):
	inputs, targets, results, lengths = result[0], result[1], result[2], result[3]
	total_words = 0
	correct_tags_with_punct = 0
	correct_tags_without_punct = 0
	num_puncts = 0

	with codecs.open("cdt2ud.txt", "w", encoding="utf-8") as fo:
		for i in range(len(inputs)):
			sent = []
			gold = []
			pred = []
			for j in range(lengths[i]):
				total_words += 1
				sent.append(mapping.index2word[inputs[i][j]])
				gold.append(mapping.index2ud_pos_tag[targets[i][j]])
				pred.append(mapping.index2ud_pos_tag[results[i][j]])

				correct_tags_with_punct += int(gold[-1] == pred[-1])
				correct_tags_without_punct += int(gold[-1] == pred[-1] and gold[-1] != "PUNCT")
				num_puncts += int(gold[-1] == "PUNCT")

			fo.write(" ".join(sent) + "\n" + " ".join(gold) + "\n" + " ".join(pred) + "\n\n")
		fo.write("With PUNCT: {}, Ignore PUNCT: {}\nCorrent PUNCTs: {}, ALL PUNCTs: {}\nTotal Words: {}".format(correct_tags_with_punct / total_words * 100, \
				(correct_tags_without_punct + num_puncts) / total_words * 100, correct_tags_with_punct - correct_tags_without_punct, num_puncts, total_words))
	

def joint_train(DataLoader_iters, r):
	# print isinstance(DataLoaders[0][2], Iterable)
	# print isinstance(DataLoaders[0][2], Iterator)
	n_iters = max(DataLoader_iters[0][1].data_size // DataLoader_iters[0][7], DataLoader_iters[1][1].data_size // DataLoader_iters[1][7]) + 1
	epochs = 10

	print_every = 100

	best_cdt2ud_accuracy = 0
	best_cdt2ud_accuracy_epoch = 0

	best_ud_dev_acc = 0
	best_ud_dev_acc_epoch = 0

	best_cdt_dev_acc = 0
	best_cdt_dev_acc_epoch = 0

	start = time.time()

	for n_iter in range(1, n_iters * epochs + 1):
		lstm_model.train()
		lstm_model.zero_grad()

		loss = 0
		for i in range(2):
			try:
				sample_batched = next(DataLoader_iters[i][3])
			except StopIteration:
				DataLoader_iters[i][3] = iter(DataLoader_iters[i][2])
				sample_batched = next(DataLoader_iters[i][3])
			inputs, targets, lengths = trainExamples(sample_batched)

			log_probs = lstm_model(inputs, DataLoader_iters[i][0])

			out = log_probs.view(-1, log_probs.size()[-1])
			target = targets.view(-1)

			if i == 0:
				loss = 1 / (1 + r) * criterion(out, target) # NLLLoss: input(2D)
			else:
				loss += r / (1 + r) * criterion(out, target) # NLLLoss: input(2D)

		loss.backward()

		optimizer.step()

		if n_iter % print_every == 0:
			print "%s (iters: %d %d%%) %.4f" % (timeSince(start), n_iter, (n_iter % n_iters) / n_iters * 100, loss.data[0])

		if n_iter % n_iters == 0:
			result, cdt2ud_accuracy = evaluate(CDT2UD_Gold_Dataloader, True)

			if cdt2ud_accuracy > best_cdt2ud_accuracy:
				best_cdt2ud_accuracy = cdt2ud_accuracy
				best_cdt2ud_accuracy_epoch = n_iter // n_iters
				output_result(result)

			print 'epochs: %d %d%% \nCDT2UD: cdt2ud_acc: %.2f%% best_cdt2ud_acc: %.2f%%(epoch=%d)' % \
				(n_iter // n_iters, n_iter / (n_iters * epochs) * 100, cdt2ud_accuracy * 100, best_cdt2ud_accuracy * 100, best_cdt2ud_accuracy_epoch)

			_, ud_dev_accuracy = evaluate(DataLoader_iters[0][4], DataLoader_iters[0][0])
			_, ud_train_accuracy = evaluate(DataLoader_iters[0][2], DataLoader_iters[0][0])

			_, cdt_dev_accuracy = evaluate(DataLoader_iters[1][4], DataLoader_iters[1][0])
			_, cdt_train_accuracy = evaluate(DataLoader_iters[1][2], DataLoader_iters[1][0])

			if ud_dev_accuracy > best_ud_dev_acc:
				best_ud_dev_acc = ud_dev_accuracy
				best_ud_dev_acc_epoch = n_iter // n_iters

			if cdt_dev_accuracy > best_cdt_dev_acc:
				best_cdt_dev_acc = cdt_dev_accuracy
				best_cdt_dev_acc_epoch = n_iter // n_iters

			print 'UD: train_acc: %.2f%% dev_acc: %.2f%% best_dev_acc: %.2f%%(epoch=%d)' % \
				(ud_train_accuracy * 100, ud_dev_accuracy * 100, best_ud_dev_acc * 100, best_ud_dev_acc_epoch)
			print 'CDT: train_acc: %.2f%% dev_acc: %.2f%% best_dev_acc: %.2f%%(epoch=%d)' % \
				(cdt_train_accuracy * 100, cdt_dev_accuracy * 100, best_cdt_dev_acc * 100, best_cdt_dev_acc_epoch)




def train(inputs, targets, lengths, UD=True):
	lstm_model.train()

	lstm_model.zero_grad()

	log_probs = lstm_model(inputs, UD)

	out = log_probs.view(-1, log_probs.size()[-1])
	target = targets.view(-1)

	loss = criterion(out, target) # NLLLoss: input(2D)

	loss.backward()

	optimizer.step()

	# for para in lstm_model.parameters():
	# 	print para

	return loss.data[0]

def evaluate(DataLoader, UD=True):
	lstm_model.eval()

	count_corrects = 0
	total_elements = 0

	total_inputs = []
	total_targets = []
	total_results = []
	total_lengths = []
	for i_batch, samples in enumerate(DataLoader):
		inputs, targets, lengths = trainExamples(samples)

		total_inputs.extend(inputs.data.tolist())
		total_targets.extend(targets.data.tolist())
		total_lengths.extend(lengths.data.tolist())

		# calculate
		log_probs = lstm_model(inputs, UD)
		# count corrects
		_, result = torch.max(log_probs, dim=2)

		total_results.extend(result.data.tolist())

		count_correct = torch.eq(result, targets)
		count_corrects += torch.sum(count_correct.data)
		total_elements += torch.sum(lengths.data)
	result = (total_inputs, total_targets, total_results, total_lengths)

	return result, count_corrects / total_elements

def pre_train(DataLoaders):

	for UD, Train_Dataset, Train_Dataloader, Dev_Dataloader, epochs, batch_size in DataLoaders:
		lstm_model.init_Hidden(batch_size)

		print_every = 100
		best_dev_accuracy = 0
		best_dev_accuracy_epoch = 0

		best_cdt2ud_accuracy = 0
		best_cdt2ud_accuracy_epoch = 0

		start = time.time()

		preface = "UD Training Start..." if UD else "CDT Training Start..." 
		print preface

		for epoch in range(1, epochs + 1):
			n_iters = Train_Dataset.data_size // batch_size
			total_loss = 0
			for i_batch, sample_batched in enumerate(Train_Dataloader):

				inputs, targets, lengths = trainExamples(sample_batched)

				loss = train(inputs, targets, lengths, UD)
				total_loss += loss
				
				if (i_batch + 1) % print_every == 0:
					print "%s (iters: %d %d%%) %.4f" % (timeSince(start), i_batch + 1, (i_batch + 1) / n_iters * 100, loss)
				# exit(0)

			result, cdt2ud_accuracy = evaluate(CDT2UD_Gold_Dataloader, True)
			_, dev_accuracy = evaluate(Dev_Dataloader, UD)
			_, train_accuracy = evaluate(Train_Dataloader, UD)

			# output_result(result)

			if cdt2ud_accuracy > best_cdt2ud_accuracy:
				best_cdt2ud_accuracy = cdt2ud_accuracy
				best_cdt2ud_accuracy_epoch = epoch

			if dev_accuracy > best_dev_accuracy:
				best_dev_accuracy = dev_accuracy
				best_dev_accuracy_epoch = epoch

			print '%s (epochs: %d %d%%) loss: %.4f (train_acc: %.2f%% dev_acc: %.2f%% best_dev_acc: %.2f%%(epoch=%d)) (cdt2ud_acc: %.2f%% best_cdt2ud_acc: %.2f%%(epoch=%d))' % \
				(timeSince(start), epoch, epoch / epochs * 100, total_loss / n_iters, train_accuracy * 100, dev_accuracy * 100, \
					best_dev_accuracy * 100, best_dev_accuracy_epoch, cdt2ud_accuracy * 100, best_cdt2ud_accuracy * 100, best_cdt2ud_accuracy_epoch)


######################################################################
# UD - UD: 	Tensorflow	93.62%	Char_Based_BiLSTM_Model (batch=1)
#			UDPipe		91.53%	Averaged_Perceptron
#			Pytorch		81.30%	Word_Based_BiLSTM_Model (input=100, batch=20, hidden=100)
#						82.51%	Word_Based_BiLSTM_Model (input=100, batch=10, hidden=100)
#						82.93%	Word_Based_BiLSTM_Model (input=100, batch=1, hidden=100)
#						83.42%	Word_Based_BiLSTM_Model (input=100, batch=10, hidden=200)
#						83.00%	Word_Based_BiLSTM_Model (input=100, batch=10, hidden=300)
#						82.58%	Word_Based_BiLSTM_Model (input=100, batch=16, hidden=200)
#						82.86%	Word_Based_BiLSTM_Model (input=100, batch=8, hidden=200)
#						84.41%	Word_Based_BiLSTM_Model (input=200, batch=10, hidden=200)
#						85.79%	Word_Based_BiLSTM_Model (input=300, batch=10, hidden=200)
#						86.69%	Word_Based_BiLSTM_Model (input=300, batch=10, hidden=200, momentum=0.9)
#						87.06%	Word_Based_BiLSTM_Model (input=300, batch=10, hidden=300, momentum=0.9)
#						87.03%	Word_Based_BiLSTM_Model (input=300, batch=1, hidden=300, momentum=0.9)
#						86.40%	Word_Based_BiLSTM_Model (input=300, batch=10, hidden=300, Adam(lr=0.001))
#						85.57%	Word_Based_BiLSTM_Model (input=300, batch=10, hidden=300, Adam(lr=0.01))
#						87.89%	Word_Based_BiLSTM_Model (input=300, batch=10, hidden=300, Adam(lr=0.005))
#						87.52%	Word_Based_BiLSTM_Model (input=300, batch=10, hidden=300, Adam(lr=0.005), dropout=0.5)
#						87.70%	Word_Based_BiLSTM_Model (input=300, batch=10, hidden=300, Adam(lr=0.005), dropout=0.9)

######################################################################
# parameters
epochs = 20
batch_size = 10
input_size = 300
hidden_size = 300
output_size_cdt = 29
output_size_ud = 17

learning_rate = 0.005


# UD Data
######################################################################
# ud training data
mapping = Map()
ud_train_data = mapping.read_from_conllu_file("/users3/dcteng/work/cdt-ud/data/ud-simplified/zhs-ud-train.conllu")
UD_Train_Dataset = ConlluPOSDataset(ud_train_data, "/users3/dcteng/work/cdt-ud/data/ud-simplified/")
UD_Train_Dataloader = DataLoader(UD_Train_Dataset, batch_size=batch_size, shuffle=True)

# ud dev data
ud_dev_data = mapping.read_from_conllu_file("/users3/dcteng/work/cdt-ud/data/ud-simplified/zhs-ud-dev.conllu")
UD_Dev_Dataset = ConlluPOSDataset(ud_dev_data, "/users3/dcteng/work/cdt-ud/data/ud-simplified/")
UD_Dev_Dataloader = DataLoader(UD_Dev_Dataset, batch_size=batch_size)


# CDT Data
######################################################################
# cdt training data
cdt_train_data = mapping.read_from_conllu_file("/users3/dcteng/work/cdt-ud/data/cdt/dep/cdt-train.conll", UD_Data=False)
CDT_Train_Dataset = ConlluPOSDataset(cdt_train_data, "/users3/dcteng/work/cdt-ud/data/cdt/dep/")
CDT_Train_Dataloader = DataLoader(CDT_Train_Dataset, batch_size=batch_size, shuffle=True)

# cdt dev data
cdt_dev_data = mapping.read_from_conllu_file("/users3/dcteng/work/cdt-ud/data/cdt/dep/cdt-holdout.conll", UD_Data=False)
CDT_Dev_Dataset = ConlluPOSDataset(cdt_dev_data, "/users3/dcteng/work/cdt-ud/data/cdt/dep/")
CDT_Dev_Dataloader = DataLoader(CDT_Dev_Dataset, batch_size=batch_size)

# cdt test data
cdt_test_data = mapping.read_from_conllu_file("/users3/dcteng/work/cdt-ud/data/cdt/dep/cdt-test.conll", UD_Data=False)
CDT_Test_Dataset = ConlluPOSDataset(cdt_test_data, "/users3/dcteng/work/cdt-ud/data/cdt/dep/")
CDT_Test_Dataloader = DataLoader(CDT_Test_Dataset, batch_size=batch_size)

# cdt2ud Data
######################################################################
# cdt2ud gold data
cdt2ud_gold_data = mapping.read_from_conllu_file("/users3/dcteng/work/cdt-ud/data/cdt2ud/gold/CDT.sample250.bigram_adapted.conll.TengDeChuan.trees.conll10", UD_Data=True)
CDT2UD_Gold_Dataset = ConlluPOSDataset(cdt2ud_gold_data, "/users3/dcteng/work/cdt-ud/data/cdt/dep/")
CDT2UD_Gold_Dataloader = DataLoader(CDT2UD_Gold_Dataset , batch_size=batch_size)


# Train
######################################################################
# loss function
criterion = nn.NLLLoss(ignore_index=100)

# model
lstm_model = MultiTaskBiLSTM(vocab_size=mapping.num_words, batch_size=batch_size, input_size=input_size, hidden_size=hidden_size, output_size_cdt=output_size_cdt, output_size_ud=output_size_ud)
lstm_model = lstm_model.cuda()

# optimization
# optimizer = optim.SGD(lstm_model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)


# print_every = 100
# best_dev_accuracy = 0
# best_dev_accuracy_epoch = 0

DataLoaders = [	(True, UD_Train_Dataset, UD_Train_Dataloader, UD_Dev_Dataloader, epochs, batch_size), 
				(False, CDT_Train_Dataset, CDT_Train_Dataloader, CDT_Dev_Dataloader, epochs, batch_size) ]

# DataLoaders[0], DataLoaders[1] = DataLoaders[1], DataLoaders[0]

# Pre_Train
######################################################################
# result GPU04	cdt2ud 				UD 			CDT
# UD 			63.33%(64.22%)		87.83%
# UD + CDT 		62.57%(72.10%)					95.09%
# CDT 			3.25%(4.88%)					95.30%
# CDT + UD		77.12%(84.03%)		88.92%                                                                                                                                                                                                                                                 

# pre_train(DataLoaders)

DataLoader_iters = [[True, UD_Train_Dataset, UD_Train_Dataloader, iter(UD_Train_Dataloader), UD_Dev_Dataloader, iter(UD_Dev_Dataloader), epochs, batch_size], 
					[False, CDT_Train_Dataset, CDT_Train_Dataloader, iter(CDT_Train_Dataloader), CDT_Dev_Dataloader, iter(CDT_Dev_Dataloader), epochs, batch_size]]

# Joint_Train
######################################################################
# parameters GPU06
# loss = 1 / (1 + r) * UDLoss + r / (1 + r) * CDTLoss
# CDT - UD 		74.00%(80.08%)			r=0.1
# 		 		87.11%(87.71%)			r=1
#				87.14%(87.73%)			r=10

r = 1

joint_train(DataLoader_iters, r)


