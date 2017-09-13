import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CRF import CRF
DROP_OUT = 0.5
FEATURE_DIM = {
    'caps': 4,
    'letter_digits': 4,
    'apostrophe_ends': 2,
    'punctuations': 2,
}

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * F.elu(x, alpha)

class LSTM_CRF(nn.Module):
    
    def __init__(self, parameter):
        super(LSTM_CRF, self).__init__()
        self.embedding_dim = parameter['embedding_dim']
        self.hidden_dim = parameter['hidden_dim']
        self.tagset_size = parameter['tagset_size']

        # Ignore hand engineer now
        #self.lstm = nn.LSTM(self.embedding_dim + sum(FEATURE_DIM.values()), 
        self.lstm = nn.LSTM(self.embedding_dim,
                self.hidden_dim, num_layers=1, batch_first = True)
        
        # Maps the output of the LSTM into tag space.
        # We add 2 here, because of START_TAG and STOP_TAG
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size+2)
        self.CRF = CRF(self.tagset_size)

    def _get_lstm_features(self, embeds):
        # batch_size * max_length * hidden_dim
        lstm_out, _ = self.lstm(embeds)
        # batch_size * max_length * (tagset_size+2)
        lstm_feats = self.hidden2tag(lstm_out)
        lstm_feats = selu(lstm_feats)
        return lstm_feats

    def get_loss(self, embeds, lens, labels):
        # Get the emission scores from the LSTM
        feats = self._get_lstm_features(embeds)
        loss =  self.CRF.get_neg_log_likilihood_loss(feats, labels, lens)

        return loss


    def forward(self, embeds): # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        feats = self._get_lstm_features(embeds)
        return self.CRF.forward(feats)


    def get_tags(self, embeds, lens):
        
        feats = self._get_lstm_features(embeds)
        _, preds = self.CRF.viterbi_decode(feats, lens)

        preds = [pred[:l].tolist() for pred, l in zip(preds.data, lens.data)]
        
        return preds


    def hand_engineer_concat(self, sentences, embeds):

        def hand_engineer_concat_kernel(sentences, embeds, keyword, num_dim):
            # batch_size * max_length      
            hand_feature = torch.LongTensor(sentences2padded(sentences, keyword)).contiguous()
            batch_size, max_length = hand_feature.size()
            # (batch_size * max_length) * CAP_DIM     
            input_hand_feature = torch.FloatTensor(batch_size * max_length, num_dim)      
            input_hand_feature.zero_()        
            # batch_size * max_length * CAP_DIM       
            input_hand_feature = input_hand_feature.scatter_(1, 
                hand_feature.view(-1, 1), 1).view(hand_feature.size()[0], hand_feature.size()[1], num_dim)        
            input_hand_feature = autograd.Variable(input_hand_feature)        
            embeds = torch.cat((embeds, input_hand_feature), 2)

            return embeds

        for input_features_name in ('caps', 'letter_digits',
                                       'apostrophe_ends','punctuations'):
            embeds = hand_engineer_concat_kernel(sentences, embeds, 
                                input_features_name, FEATURE_DIM[input_features_name])

        return embeds
