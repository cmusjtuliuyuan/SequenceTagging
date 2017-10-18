import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import LstmCrfModel
from utils import sentences2padded, char2padded, get_lens, sequence_mask
from loader import FEATURE_DIM

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

def normalize_log_distribution(log_distribution, dim):
    """
    Arguments:
        log_distribution: [batch_size, max_length, n_labels] FloatTensor
        dim: int
    Return:
        log_distribution_normalized: [batch_size, max_length, n_labels] FloatTensor
    """


    log_distribution_min, _ = torch.min(log_distribution, dim, keepdim=True)
    log_distribution = log_distribution - log_distribution_min
    distribution = torch.exp(log_distribution)
    distribution_sum = torch.sum(distribution, dim, keepdim=True)
    distribution_normalized = distribution / distribution_sum
    log_distribution_normalized = torch.log(distribution_normalized)
    return log_distribution_normalized

class Autoencoder(nn.Module):
    
    def __init__(self, parameter):
        super(Autoencoder, self).__init__()
        self.embedding_dim = parameter['embedding_dim']
        self.vocab_size = parameter['vocab_size']
        self.char_dim = parameter['char_dim']
        self.char_lstm_dim = parameter['char_lstm_dim']
        self.tagset_size = parameter['tagset_size']
        self.freeze = parameter['freeze']
        self.is_cuda = parameter['cuda']
        self.char_size = parameter['char_size']
        
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.cap_embeds = nn.Embedding(FEATURE_DIM['caps'], FEATURE_DIM['caps'])
        self.letter_digits_embeds = nn.Embedding(FEATURE_DIM['letter_digits'], FEATURE_DIM['letter_digits'])
        self.apostrophe_ends_embeds = nn.Embedding(FEATURE_DIM['apostrophe_ends'], FEATURE_DIM['apostrophe_ends'])
        self.punctuations_embeds = nn.Embedding(FEATURE_DIM['punctuations'], FEATURE_DIM['punctuations'])

        self.encoder = LstmCrfModel.LSTM_CRF(parameter)
        #self.decoder = nn.LSTM(self.tagset_size, self.vocab_size,
        #                    num_layers=1, batch_first = True)
        self.decoder = nn.Linear(self.tagset_size, self.vocab_size)
        self.loss_function = nn.CrossEntropyLoss(ignore_index = self.vocab_size+1)
        if self.char_dim!=0:
            self.char_embeds = nn.Embedding(self.char_size, self.char_dim)
            self.char_lstm_forward = nn.LSTM(self.char_dim, self.char_lstm_dim,
                            num_layers=1, batch_first = False)
            self.char_lstm_backward = nn.LSTM(self.char_dim, self.char_lstm_dim,
                            num_layers=1, batch_first = False)


    def init_word_embedding(self, init_matrix):
        if self.is_cuda:
            self.word_embeds.weight=nn.Parameter(torch.FloatTensor(init_matrix).cuda())
        else:
            self.word_embeds.weight=nn.Parameter(torch.FloatTensor(init_matrix))
        self.word_embeds.weight.requires_grad = not self.freeze

    def get_embeds_chars(self, sentences):
        forward_chars, backward_chars, lens_chars = char2padded(sentences)

        fc_lstm_array=[]
        bc_lstm_array=[]
        for fc, bc, lc in zip(forward_chars, backward_chars, lens_chars):
            input_fc = autograd.Variable(torch.LongTensor(fc)).cuda()
            input_bc = autograd.Variable(torch.LongTensor(bc)).cuda()
            input_lc = autograd.Variable(torch.LongTensor(lc)).cuda()
            batch_size = input_lc.size()[0]
            input_lc = ((input_lc - 1)<0).type(torch.LongTensor)+input_lc - 1
            input_lc = input_lc.unsqueeze(-1).unsqueeze(-1).expand(batch_size,1,self.char_lstm_dim)

            if self.is_cuda:
                input_fc = input_fc.cuda()
                input_bc = input_bc.cuda()
                input_lc = input_lc.cuda()
            
            fc_embeds = self.char_embeds(input_fc)
            bc_embeds = self.char_embeds(input_bc)
            fc_lstm_out, _ = self.char_lstm_forward(fc_embeds)
            bc_lstm_out, _ = self.char_lstm_backward(bc_embeds)
            
            fc_lstm_out = fc_lstm_out.gather(dim=1, index = input_lc)
            bc_lstm_out = bc_lstm_out.gather(dim=1, index = input_lc)
            fc_lstm_array.append(fc_lstm_out)
            bc_lstm_array.append(bc_lstm_out)
        fc_lstm = torch.cat(fc_lstm_array, dim=1)
        bc_lstm = torch.cat(bc_lstm_array, dim=1)
        embeds_chars = torch.cat((fc_lstm, bc_lstm), dim=2)
        return embeds_chars


    def get_embeds(self, sentences):
        input_words = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'words')))
        input_caps = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'caps')))
        input_letter_digits = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'letter_digits')))
        input_apostrophe_ends = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'apostrophe_ends')))
        input_punctuations = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'punctuations')))
        lens = autograd.Variable(torch.LongTensor(get_lens(sentences, 'words')))
        if self.is_cuda:
            input_words = input_words.cuda()
            input_caps = input_caps.cuda()
            input_letter_digits = input_letter_digits.cuda()
            input_apostrophe_ends = input_apostrophe_ends.cuda()
            input_punctuations = input_punctuations.cuda()
            lens = lens.cuda()
        # batch_size * max_length * embedding_dim
        embeds_word = self.word_embeds(input_words)
        #embeds.register_hook(save_grad('embeds'))
        embeds_caps = self.cap_embeds(input_caps)
        embeds_letter_digits = self.letter_digits_embeds(input_letter_digits)
        embeds_apostrophe_ends = self.apostrophe_ends_embeds(input_apostrophe_ends)
        embeds_punctuations = self.punctuations_embeds(input_punctuations)
        embeds_chars = self.get_embeds_chars(sentences)
        embeds = torch.cat((embeds_word, embeds_caps, embeds_letter_digits,
                        embeds_apostrophe_ends, embeds_punctuations, embeds_chars),2)
        return embeds, lens

    def get_loss_supervised(self, sentences): # supervised loss
        # batch_size * max_length
        labels = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'tags')))
        if self.is_cuda:
            labels = labels.cuda()  

        embeds, lens = self.get_embeds(sentences)
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
            
            loss = self.loss_function(decoder_out.contiguous().view(-1, self.vocab_size),
                                        input_words.contiguous().view(-1))
            #print loss.data.cpu().numpy()
            return loss, True
        return None, False
        '''
        lens = autograd.Variable(torch.LongTensor(get_lens(sentences, 'words')))
        if self.is_cuda:
            lens = lens.cuda()
        decoder_out, embeds = self.forward(sentences)
        batch_size, max_length, embed_dim = embeds.size()

        if max_length < 80:
            mask = sequence_mask(lens, cuda = self.is_cuda).float().unsqueeze(-1).expand_as(decoder_out)

            loss_matrix = mask * (decoder_out - embeds)

            loss = 2*torch.sum(loss_matrix*loss_matrix)/(batch_size*max_length*embed_dim)

            return loss, True
        return None, False
        '''


    def forward(self, sentences):

        embeds, lens = self.get_embeds(sentences)
        # batch_size * max_length * tagset_size+2
        encoder_out = self.encoder.forward(embeds, lens)

        encoder_out_normalized = normalize_log_distribution(encoder_out[:,:,:self.tagset_size], dim=2)

        # batch_size * max_length * embedding_dim
        #decoder_out, _ = self.decoder.forward(encoder_out_normalized)
        decoder_out = self.decoder.forward(encoder_out_normalized)
        return decoder_out, embeds


    def get_tags(self, sentences):

        embeds, lens = self.get_embeds(sentences)
        preds = self.encoder.get_tags(embeds, lens)

        return preds
