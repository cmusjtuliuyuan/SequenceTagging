import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import LstmCrfModel
from utils import sentences2padded, get_lens, sequence_mask

class Autoencoder(nn.Module):
    
    def __init__(self, parameter):
        super(Autoencoder, self).__init__()
        self.embedding_dim = parameter['embedding_dim']
        self.vocab_size = parameter['vocab_size']
        # +2 because of START_TAG STOP_TAG
        self.tagset_size = parameter['tagset_size'] + 2
        self.freeze = parameter['freeze']
        self.is_cuda = parameter['cuda']
        
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.encoder = LstmCrfModel.LSTM_CRF(parameter)
        self.decoder = nn.LSTM(self.tagset_size, self.vocab_size,
                            num_layers=1, batch_first = True)
        self.loss_function = nn.NLLLoss(ignore_index = self.vocab_size+1)


    def init_word_embedding(self, init_matrix):
        self.word_embeds.weight=nn.Parameter(torch.FloatTensor(init_matrix))
        self.word_embeds.weight.requires_grad = not self.freeze

    def get_loss_supervised(self, sentences): # supervised loss
        # batch_size * max_length
        input_words = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'words')))
        lens = autograd.Variable(torch.LongTensor(get_lens(sentences, 'words')))
        labels = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'tags')))
        if self.is_cuda:
            input_words = input_words.cuda()
            lens = lens.cuda()
            labels = labels.cuda()  
        # batch_size * max_length * embedding_dim
        embeds = self.word_embeds(input_words)
        # Remove softmax layer
        #embeds = F.softmax(embeds.view(-1, self.embedding_dim)).view(*embeds.size())
        # Ignore hand engineer now
        #embeds = self.hand_engineer_concat(sentences, embeds)
        loss = self.encoder.get_loss(embeds, lens, labels)
        return loss, True

    def get_loss_unsupervised(self, sentences): # unsupervised loss
    
        input_words = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'words', 
                                replace = self.vocab_size+1)).contiguous())
        max_length = input_words.size()[1]
        if max_length < 80:
            if self.is_cuda:
                input_words = input_words.cuda()
            decoder_out, embeds = self.forward(sentences)
            
            loss = self.loss_function(decoder_out.contiguous().view(-1, self.vocab_size), input_words.view(-1))
            #print "one batch", max_length
            return loss, True
        return None, False
    '''
        lens = autograd.Variable(torch.LongTensor(get_lens(sentences, 'words')))
        if self.is_cuda:
            lens = lens.cuda()
        decoder_out, embeds = self.forward(sentences)
        batch_size, max_length, embed_dim = embeds.size()

        mask = sequence_mask(lens, cuda = self.is_cuda).float().unsqueeze(-1).expand_as(decoder_out)

        loss_matrix = mask * (decoder_out - embeds)

        loss = 8*torch.sum(loss_matrix*loss_matrix)/(batch_size*max_length*embed_dim)

        return loss
    '''

    def forward(self, sentences):
        # batch_size * max_length
        input_words = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'words')))
        lens = autograd.Variable(torch.LongTensor(get_lens(sentences, 'words')))
        if self.is_cuda:
            input_words = input_words.cuda()
            lens = lens.cuda()
        # batch_size * max_length * embedding_dim
        embeds = self.word_embeds(input_words)
        # batch_size * max_length * tagset_size+2
        encoder_out = self.encoder.forward(embeds, lens)
        # batch_size * max_length * embedding_dim
        decoder_out, _ = self.decoder.forward(encoder_out)

        return decoder_out, embeds


    def get_tags(self, sentences):

        input_words = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'words')))
        lens = autograd.Variable(torch.LongTensor(get_lens(sentences, 'words')))
        if self.is_cuda:
            input_words = input_words.cuda()
            lens = lens.cuda()

        embeds = self.word_embeds(input_words)
        preds = self.encoder.get_tags(embeds, lens)

        return preds
