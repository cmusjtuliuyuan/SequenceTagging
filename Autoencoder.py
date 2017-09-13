import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import LstmCrfModel
from utils import sentences2padded, get_lens

class Autoencoder(nn.Module):
    
    def __init__(self, parameter):
        super(Autoencoder, self).__init__()
        self.embedding_dim = parameter['embedding_dim']
        self.vocab_size = parameter['vocab_size']
        # +2 because of START_TAG STOP_TAG
        self.tagset_size = parameter['tagset_size'] + 2
        self.freeze = parameter['freeze']
        
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.encoder = LstmCrfModel.LSTM_CRF(parameter)
        self.decoder = nn.LSTM(self.tagset_size, self.embedding_dim,
                            num_layers=1, batch_first = True)

    def init_word_embedding(self, init_matrix):
        self.word_embeds.weight=nn.Parameter(torch.FloatTensor(init_matrix))
        self.word_embeds.weight.requires_grad = not self.freeze

    def get_loss_supervised(self, sentences): # supervised loss
        # batch_size * max_length
        input_words = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'words')))
        lens = autograd.Variable(torch.LongTensor(get_lens(sentences, 'words')))
        labels = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'tags')))  
        # batch_size * max_length * embedding_dim
        embeds = self.word_embeds(input_words)
        # Remove softmax layer
        #embeds = F.softmax(embeds.view(-1, self.embedding_dim)).view(*embeds.size())
        # Ignore hand engineer now
        #embeds = self.hand_engineer_concat(sentences, embeds)
        loss = self.encoder.get_loss(embeds, lens, labels)
        return loss

    def get_tags(self, sentences):

        input_words = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'words')))
        lens = autograd.Variable(torch.LongTensor(get_lens(sentences, 'words')))

        embeds = self.word_embeds(input_words)
        preds = self.encoder.get_tags(embeds, lens)

        return preds