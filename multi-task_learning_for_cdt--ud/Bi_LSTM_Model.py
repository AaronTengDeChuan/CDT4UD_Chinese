import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random



class MultiTaskBiLSTM(nn.Module):

	def __init__(self, vocab_size, batch_size, pretrained_embedding_size, random_embedding_size, hidden_size, output_size_cdt, output_size_ud, pretrained_embeds, dropout_prob=0.5, bidirectional=True):
		super(MultiTaskBiLSTM, self).__init__()
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.pretrained_embedding_size = pretrained_embedding_size
		self.random_embedding_size = random_embedding_size
		self.hidden_size = hidden_size
		self.output_size_cdt = output_size_cdt
		self.output_size_ud = output_size_ud

		self.bidirectional = bidirectional
		self.dropout_prob = dropout_prob

		self.w1_embeddings = nn.Embedding(self.vocab_size, self.random_embedding_size).cuda()
		self.w2_embeddings = [nn.Embedding(self.vocab_size, self.pretrained_embedding_size).cuda()]
		self.w2_embeddings[0].weight.data.copy_(pretrained_embeds)
		self.w2_embeddings[0].weight.requires_grad = False


		self.initHidden = Variable(torch.zeros(2 if self.bidirectional else 1, self.batch_size, self.hidden_size).cuda())
		self.initCell = Variable(torch.zeros(2 if self.bidirectional else 1, self.batch_size, self.hidden_size).cuda())

		self.lstm = nn.LSTM(self.pretrained_embedding_size + self.random_embedding_size, self.hidden_size, batch_first=True, bidirectional=self.bidirectional)

		self.h2o_cdt = nn.Linear((2 if self.bidirectional else 1) * self.hidden_size, self.output_size_cdt).cuda()
		self.h2o_ud = nn.Linear((2 if self.bidirectional else 1) * self.hidden_size, self.output_size_ud).cuda()

	def forward(self, inputs, UD=True):
		w1_embeds = self.w1_embeddings(inputs)
		w2_embeds = self.w2_embeddings[0](inputs)
		embeds = torch.cat((w1_embeds, w2_embeds), dim=2).cuda()
		# print (embeds.data.shape)
		# print (w1_embeds.data[:1])

		batch_size = inputs.size()[0]
		if batch_size != self.batch_size:
			self.initHidden = Variable(torch.zeros(2 if self.bidirectional else 1, batch_size, self.hidden_size).cuda())
			self.initCell = Variable(torch.zeros(2 if self.bidirectional else 1, batch_size, self.hidden_size).cuda())

		output, (hidden, cell) = self.lstm(embeds, (self.initHidden, self.initCell)) # [batch, seq_len, hidden_size * directions]
		output = F.dropout(output, p=self.dropout_prob, training=self.training)

		if batch_size != self.batch_size:
			self.initHidden = Variable(torch.zeros(2 if self.bidirectional else 1, self.batch_size, self.hidden_size).cuda())
			self.initCell = Variable(torch.zeros(2 if self.bidirectional else 1, self.batch_size, self.hidden_size).cuda())

		if UD:
			out = self.h2o_ud(output.contiguous().view(-1, output.size()[-1]))
		else:
			out = self.h2o_cdt(output.contiguous().view(-1, output.size()[-1]))

		log_probs = F.log_softmax(out).cuda().view(output.size()[0], output.size()[1], -1)

		return log_probs

	def init_Hidden(self, batch_size):
		self.batch_size = batch_size
		self.initHidden = Variable(torch.zeros(2 if self.bidirectional else 1, self.batch_size, self.hidden_size).cuda())
		self.initCell = Variable(torch.zeros(2 if self.bidirectional else 1, self.batch_size, self.hidden_size).cuda())

if __name__ == "__main__":
	from Dataset import ConlluPOSDataset, Map
	from torch.utils.data import Dataset, DataLoader

	mapping = Map()
	data = mapping.read_from_conllu_file("/users3/dcteng/work/cdt-ud/data/ud-traditional/UD_Chinese/zh-ud-train.conllu")

	UD_Dataset = ConlluPOSDataset(data, "/users3/dcteng/work/cdt-ud/data/ud-traditional/UD_Chinese/")
	UD_Dataloader = DataLoader(UD_Dataset, batch_size=5, shuffle=True)

	lstm_model = MultiTaskBiLSTM(vocab_size=mapping.num_words, batch_size=5, input_size=50, hidden_size=20, output_size_cdt=20, output_size_ud=18)

	for i_batch, sample_batched in enumerate(UD_Dataloader):
		inputs = sample_batched["sentence"]
		inputs = Variable(inputs)

		log_probs = lstm_model(inputs)
		print log_probs
		break