import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


CAP_DIM = 4
def sentences2padded(sentences, keyword):
    # Form Batch_Size * Length
    max_length = max([len(sentence[keyword]) for sentence in sentences])
    batch_size = len(sentences)
    def pad_seq(seq, max_length):
        padded_seq = seq + [0 for i in range(max_length - len(seq))]
        return padded_seq
    padded =[pad_seq(sentence[keyword], max_length) for sentence in sentences]
    return padded


class LSTMTagger(nn.Module):
    
    def __init__(self, parameter):
        super(LSTMTagger, self).__init__()
        self.lower = parameter['lower']
        self.hidden_dim = parameter['hidden_dim']
        self.tagset_size = parameter['tagset_size']
        
        self.word_embeddings = nn.Embedding(parameter['vocab_size'], 
                                            parameter['embedding_dim'])

        self.embedding_dim = parameter['embedding_dim'] + CAP_DIM
        # The LSTM takes word embeddings and captical embedding as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.embedding_dim, parameter['hidden_dim'])
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(parameter['hidden_dim'], parameter['tagset_size'])
        self.log_softmax = nn.LogSoftmax()
        self.loss_function = nn.NLLLoss()
    
    def init_word_embedding(self, init_matrix):
        self.word_embeddings.weight=nn.Parameter(torch.FloatTensor(init_matrix))

    def forward(self, sentences):
        #  max_length * batch_size
        input_words = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'words')).transpose(0,1),
                                         requires_grad = False)
        #  max_length * batch_size * embedding_dim
        embeds = self.word_embeddings(input_words)
        
        # We first need to convert captial information it into on-hot. Then concat it with the word_embedding layer
        # max_length * batch_size
        caps = torch.LongTensor(sentences2padded(sentences, 'caps')).transpose(0, 1).contiguous()
        # (max_length * batch_size) * CAP_DIM
        input_caps = torch.FloatTensor(caps.size()[0]*caps.size()[1], CAP_DIM)
        input_caps.zero_()
        # max_length * batch_size * CAP_DIM
        input_caps = input_caps.scatter_(1, caps.view(-1, 1), 1).view(caps.size()[0], caps.size()[1], CAP_DIM)
        input_caps = autograd.Variable(input_caps, requires_grad = False)
        embeds = torch.cat((embeds, input_caps), 2)

        # max_length * batch_size * hidden_dim
        lstm_out, _ = self.lstm(embeds)
        # (max_length * batch_size) * tagset_size
        tag_space = self.hidden2tag(lstm_out.view(-1, self.hidden_dim))
        tag_scores = self.log_softmax(tag_space)
        #  max_length * batch_size * tagset_size
        return tag_scores.view(-1, len(sentences), self.tagset_size)


    def get_tags(self, sentences):
        # get_tag does not support batch now
        # length * 1 * tagset_size
        predict = self.forward(sentences).view(-1, self.tagset_size)
        _, tags = torch.max(predict, dim=1)
        tags = tags.data.numpy().reshape((-1,))
        return tags
    
    def get_loss(self, sentences):

        # max_length * batch_size * tagset_size
        predict = self.forward(sentences)
        # max_length * batch_size
        target = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'tags')).transpose(0,1).contiguous(),
                                         requires_grad = False)
        loss = self.loss_function(predict.view(-1, self.tagset_size), target.view(-1))
        return loss

