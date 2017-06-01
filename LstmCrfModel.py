import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loader import CAP_DIM
from CRF import CRF
DROP_OUT = 0.5
START_TAG = -2
STOP_TAG = -1


class BiLSTM_CRF(nn.Module):
    
    def __init__(self, parameter):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = parameter['embedding_dim']
        self.hidden_dim = parameter['hidden_dim']
        self.vocab_size = parameter['vocab_size']
        self.tagset_size = parameter['tagset_size']
        self.lower = parameter['lower']
        self.decode_method = parameter['decode_method']
        self.loss_function = parameter['loss_function']
        
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        if self.lower:
            self.embedding_dim += CAP_DIM

        self.dropout = nn.Dropout(p=DROP_OUT)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim/2, num_layers=1, bidirectional=True)
        
        # Maps the output of the LSTM into tag space.
        # We add 2 here, because of START_TAG and STOP_TAG
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size+2)
        
        self.CRF = CRF(self.tagset_size)
        
        self.hidden = self.init_hidden()


    def init_word_embedding(self, init_matrix):
        self.word_embeds.weight=nn.Parameter(torch.FloatTensor(init_matrix))


    def init_hidden(self):
        return ( autograd.Variable( torch.zeros(2, 1, self.hidden_dim)),
                 autograd.Variable( torch.zeros(2, 1, self.hidden_dim)) )



    def _get_lstm_features(self, dropout, **sentence):
    	self.hidden = self.init_hidden()
        input_words = sentence['input_words']
        embeds = self.word_embeds(input_words)
        if self.lower:
              # We first need to convert it into on-hot. Then concat it with the word_embedding layer
            caps = sentence['input_caps']
            input_caps = torch.FloatTensor(len(caps), CAP_DIM)
            input_caps.zero_()
            input_caps.scatter_(1, caps.view(-1,1) ,1)
            input_caps = autograd.Variable(input_caps)
            embeds = torch.cat((embeds, input_caps),1)

        #if dropout:
        #    embeds = self.dropout(embeds)

        lstm_out, self.hidden = self.lstm(embeds.view(len(input_words), 1, -1))
        lstm_out = lstm_out.view(len(input_words), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats


    def get_loss(self, tags, **sentence):
        if self.loss_function == 'likelihood':
            return self.get_neg_log_likilihood_loss(tags, **sentence)
        elif self.loss_function == 'labelwise':
            return self.get_labelwise_loss(tags, **sentence)
        else:
            print("ERROR: The parameter of loss function is wrong")


    def get_neg_log_likilihood_loss(self, tags, **sentence):
        # nonegative log likelihood
        feats = self._get_lstm_features(dropout=False, **sentence)
        forward_score, _ = self.CRF._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score


    def get_labelwise_loss(self, tags, **sentence):
    	'''
    	Training Conditional Random Fields for Maximum Labelwise Accuracy 
    	Please look at this paper
    	'''
        # Get the emission scores from the BiLSTM
        feats = self._get_lstm_features(dropout=False, **sentence)
        
        # Get the marginal distribution
        score, _ = self.CRF._marginal_decode(feats)
        tags = tags.data.numpy()

        loss = autograd.Variable(torch.Tensor([0.]))
        Q = nn.Sigmoid()
        for tag, log_p in zip(tags, score):
            Pw = log_p[tag]
            if tag == 0:
                not_tag = log_p[1:]
            elif tag == len(log_p) - 1:
                not_tag = log_p[:tag]
            else:
                not_tag = torch.cat((log_p[:tag], log_p[tag+1:]))
            maxPw = torch.max(not_tag)
            loss = loss - Q(Pw - maxPw)
        return loss


    def forward(self, **sentence): # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        feats = self._get_lstm_features(dropout=False, **sentence)
        
        # Find the best path, given the features.
        if self.decode_method == 'marginal':
            score, tag_seq = self.CRF._marginal_decode(feats)
        elif self.decode_method == 'viterbi':
            score, tag_seq = self.CRF._viterbi_decode(feats)
        else:
            print("Error wrong decode method")

        return score, tag_seq


    def get_tags(self, **sentence):
        score, tag_seq = self.forward(**sentence)
        return np.asarray(tag_seq).reshape((-1,))
