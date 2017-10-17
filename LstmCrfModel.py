import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as R
import numpy as np
from CRF import CRF
DROP_OUT = 0.5
from loader import FEATURE_DIM

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)

class LSTM_CRF(nn.Module):
    
    def __init__(self, parameter):
        super(LSTM_CRF, self).__init__()
        self.embedding_dim = parameter['embedding_dim']+sum(FEATURE_DIM.values())
        self.hidden_dim = parameter['hidden_dim']
        self.tagset_size = parameter['tagset_size']

        self.embeds_input = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim,
                self.hidden_dim/2, num_layers = 2, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(p=DROP_OUT)
        # Maps the output of the LSTM into tag space.
        # We add 2 here, because of START_TAG and STOP_TAG
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size+2)
        self.CRF = CRF(self.tagset_size, parameter)


    def _get_lstm_features(self, embeds, lens):
        embeds = self.embeds_input(embeds)
        embeds = selu(embeds.view(-1, self.embedding_dim)).view(*embeds.size())

        # LSTM part:
        embeds_packed = R.pack_padded_sequence(embeds, lens.data.tolist(),
                                          batch_first=True)
        lstm_out, _ = self.lstm(embeds_packed)
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)

        # batch_size * max_length * (tagset_size+2)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        lstm_feats = selu(lstm_feats.view(-1, self.tagset_size+2)).view(*lstm_feats.size())

        return lstm_feats

    def get_loss(self, embeds, lens, labels):
        # Get the emission scores from the LSTM
        feats = self._get_lstm_features(embeds, lens)
        loss =  self.CRF.get_neg_log_likilihood_loss(feats, labels, lens)

        return loss


    def forward(self, embeds, lens): # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        feats = self._get_lstm_features(embeds, lens)
        log_distribution = self.CRF.forward(feats, lens)
        return log_distribution


    def get_tags(self, embeds, lens):
        
        feats = self._get_lstm_features(embeds, lens)
        _, preds = self.CRF.marginal_decode(feats, lens)

        preds = [pred[:l].tolist() for pred, l in zip(preds.data, lens.data)]
        
        return preds
