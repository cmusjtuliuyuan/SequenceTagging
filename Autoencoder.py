import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
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
        self.letter_digits_embeds = nn.Embedding(FEATURE_DIM['letter_digits'],
                                                 FEATURE_DIM['letter_digits'])
        I.xavier_normal(self.word_embeds.weight.data)
        I.xavier_normal(self.cap_embeds.weight.data)
        I.xavier_normal(self.letter_digits_embeds.weight.data)

        self.encoder = LstmCrfModel.LSTM_CRF(parameter)
        #self.decoder = nn.LSTM(self.tagset_size, self.vocab_size,
        #                    num_layers=1, batch_first = True)
        self.decoder = nn.Linear(self.tagset_size, self.vocab_size)
        self.loss_function = nn.CrossEntropyLoss(ignore_index = self.vocab_size+1)
        self.embeds_parameters = list(self.word_embeds.parameters())\
                                + list(self.cap_embeds.parameters())\
                                + list(self.letter_digits_embeds.parameters())
        if self.char_dim!=0:
            self.char_embeds = nn.Embedding(self.char_size, self.char_dim)
            self.char_lstm_forward = nn.LSTM(self.char_dim, self.char_lstm_dim,
                            num_layers=1, batch_first = False)
            self.char_lstm_backward = nn.LSTM(self.char_dim, self.char_lstm_dim,
                            num_layers=1, batch_first = False)
            self.embeds_parameters += list(self.char_embeds.parameters())\
                                + list(self.char_lstm_forward.parameters())\
                                + list(self.char_lstm_backward.parameters())
            I.xavier_normal(self.char_embeds.weight.data)


    def init_word_embedding(self, init_matrix):
        if self.is_cuda:
            self.word_embeds.weight=nn.Parameter(torch.FloatTensor(init_matrix).cuda())
        else:
            self.word_embeds.weight=nn.Parameter(torch.FloatTensor(init_matrix))
        self.word_embeds.weight.requires_grad = not self.freeze


    def get_embeds_chars(self, sentences):
        batch_size = len(sentences)
        forward_chars, backward_chars, lens_chars, max_char_length = char2padded(sentences)
        
        fc_input_array=[]
        bc_input_array=[]
        lc_input_array=[]
        for fc, bc, lc in zip(forward_chars, backward_chars, lens_chars):
            input_fc = autograd.Variable(torch.LongTensor(fc))
            input_bc = autograd.Variable(torch.LongTensor(bc))
            input_lc = autograd.Variable(torch.LongTensor(lc))
            fc_input_array.append(input_fc.unsqueeze(dim=1))
            bc_input_array.append(input_bc.unsqueeze(dim=1))
            lc_input_array.append(input_lc.unsqueeze(dim=1))
        # batch_size * sentence_length, char_length
        fc_input_cat = torch.cat(fc_input_array, dim=1).contiguous().view(-1,max_char_length)
        bc_input_cat = torch.cat(bc_input_array, dim=1).contiguous().view(-1,max_char_length)
        lc_input_cat = torch.cat(lc_input_array, dim=1).contiguous().view(-1)
        # batch_size * sentence_length, 1, char_embed_dim
        lc_index = lc_input_cat.unsqueeze(-1).unsqueeze(-1).expand(fc_input_cat.size()[0],1,self.char_lstm_dim)
        lc_index = ((lc_index - 1)<0).type(torch.LongTensor)+lc_index - 1

        if self.is_cuda:
            fc_input_cat = fc_input_cat.cuda()
            bc_input_cat = bc_input_cat.cuda()
            lc_index = lc_index.cuda()

        # batch_size * sentence_length, char_length, char_embed_dim
        fc_embeds = self.char_embeds(fc_input_cat)
        bc_embeds = self.char_embeds(bc_input_cat)
        # batch_size * sentence_length, char_length, char_lstm_dim
        fc_lstm_out, _ = self.char_lstm_forward(fc_embeds)
        bc_lstm_out, _ = self.char_lstm_backward(bc_embeds)
        fc_lstm_out = fc_lstm_out.gather(dim=1, index = lc_index)\
                            .unsqueeze(dim=1).view(batch_size,-1, self.char_lstm_dim)
        bc_lstm_out = bc_lstm_out.gather(dim=1, index = lc_index)\
                            .unsqueeze(dim=1).view(batch_size,-1, self.char_lstm_dim)

        embeds_chars = torch.cat((fc_lstm_out, bc_lstm_out), dim=2)
        return embeds_chars


    def get_embeds(self, sentences):
        input_words = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'words')))
        input_caps = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'caps')))
        input_letter_digits = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'letter_digits')))
        lens = autograd.Variable(torch.LongTensor(get_lens(sentences, 'words')))
        if self.is_cuda:
            input_words = input_words.cuda()
            input_caps = input_caps.cuda()
            input_letter_digits = input_letter_digits.cuda()
            lens = lens.cuda()
        # batch_size * max_length * embedding_dim
        embeds_word = self.word_embeds(input_words)
        #embeds.register_hook(save_grad('embeds'))
        embeds_caps = self.cap_embeds(input_caps)
        embeds_letter_digits = self.letter_digits_embeds(input_letter_digits)
        embeds_list = [embeds_word, embeds_caps, embeds_letter_digits]
        if self.char_dim!=0:
            embeds_list.append(self.get_embeds_chars(sentences))
        embeds = torch.cat((embeds_list),2)
        return embeds, lens

    def get_loss_supervised(self, sentences): # supervised loss
        # batch_size * max_length
        labels = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'tags')))
        if self.is_cuda:
            labels = labels.cuda()  

        embeds, lens = self.get_embeds(sentences)
        loss = self.encoder.get_loss(embeds, lens, labels)
        return loss

    def get_loss_unsupervised(self, sentences): # unsupervised loss

        input_words = autograd.Variable(torch.LongTensor(sentences2padded(sentences, 'words', 
                                replace = self.vocab_size+1)).contiguous())
        max_length = input_words.size()[1]
        if self.is_cuda:
            input_words = input_words.cuda()
        decoder_out, embeds = self.forward(sentences)
        
        loss = self.loss_function(decoder_out.contiguous().view(-1, self.vocab_size),
                                    input_words.contiguous().view(-1))
        #print loss.data.cpu().numpy()
        return loss
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
