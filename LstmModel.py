import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


CAP_DIM = 4
def sentences2padded(sentences, keyword, replace = 0):
    # Form Batch_Size * Length
    max_length = max([len(sentence[keyword]) for sentence in sentences])
    def pad_seq(seq, max_length):
        padded_seq = seq + [replace for i in range(max_length - len(seq))]
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

        self.embedding_dim = parameter['embedding_dim']
        # The LSTM takes word embeddings and captical embedding as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size = self.embedding_dim, hidden_size = parameter['hidden_dim'],
                    batch_first = True)
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(parameter['hidden_dim'], parameter['tagset_size'])
        #self.log_softmax = nn.LogSoftmax()
        self.loss_function = nn.NLLLoss(ignore_index = self.tagset_size+1)
    
    def init_word_embedding(self, init_matrix):
        self.word_embeddings.weight=nn.Parameter(torch.FloatTensor(init_matrix))

    def forward(self, sentences):
        # batch_size * max_length
        input_words = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'words')))
        batch_size = input_words.size()[0]
        max_length = input_words.size()[1]
        # batch_size * max_length * embedding_dim
        embeds = self.word_embeddings(input_words)
        '''
        ##Just Ignore this part.
        # We first need to convert captial information it into on-hot. Then concat it with the word_embedding layer
        # max_length * batch_size
        caps = torch.LongTensor(sentences2padded(sentences, 'caps')).contiguous()
        # (batch_size * max_length) * CAP_DIM
        input_caps = torch.FloatTensor(batch_size * max_length, CAP_DIM)
        input_caps.zero_()
        # batch_size * max_length * CAP_DIM
        input_caps = input_caps.scatter_(1, caps.view(-1, 1), 1).view(caps.size()[0], caps.size()[1], CAP_DIM)
        input_caps = autograd.Variable(input_caps)
        embeds = torch.cat((embeds, input_caps), 2)
        '''
        # batch_size * max_length * hidden_dim
        lstm_out, _ = self.lstm(embeds)
        # batch_size * max_length * tagset_size
        tag_space = self.hidden2tag(lstm_out)
        '''
        ##TODO don't know to choose which one. log_softmax is so confusing!!
        # tagset_size * (batch_size * max_length)
        tag_scores = self.log_softmax(tag_space.view(batch_size*max_length, 
                            self.tagset_size).transpose(0,1))
        #  batch_size * max_length * tagset_size
        return tag_scores.transpose(0,1).contiguous().view(batch_size, max_length, self.tagset_size)
        '''
        # (batch_size * max_length) * tagset_size
        tag_scores = F.log_softmax(tag_space.view(-1, self.tagset_size))
        return tag_scores.view(batch_size, max_length, self.tagset_size)
        

    def get_tags(self, sentences):
        # get_tag does not support batch now
        # length * tagset_size
        predict = self.forward(sentences).view(-1, self.tagset_size)
        _, tags = torch.max(predict, dim=1)
        tags = tags.data.numpy().reshape((-1,))
        return tags
    
    def get_loss(self, sentences):

        # batch_size * max_length * tagset_size
        predict = self.forward(sentences)
        # batch_size * max_length
        target = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'tags',
                            replace = self.tagset_size+1)).contiguous())
        loss = self.loss_function(predict.view(-1, self.tagset_size), target.view(-1))
        return loss

