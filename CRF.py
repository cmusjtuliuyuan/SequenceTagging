import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import sequence_mask

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec, dim=0):
    max, idx = torch.max(vec, dim, keepdim=True)
    max_exp = max.expand_as(vec)
    return max.squeeze(-1) + torch.log(torch.sum(torch.exp(vec - max_exp), dim))

class CRF(nn.Module):
    '''
    Thanks, kaniblu!
    '''
    def __init__(self, tagset_size):
        super(CRF, self).__init__()
        # We add 2 here, because of START_TAG and STOP_TAG
        self.tagset_size = tagset_size + 2
        self.START_TAG = tagset_size
        self.STOP_TAG = tagset_size + 1
        # transition[i,j] means from j to i
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.START_TAG, :] = -10000.
        self.transitions.data[:, self.STOP_TAG] = -10000.

    def _forward_alg(self, logits, lens):
        '''
        Arguments:
            logits: [batch_size, max_length, tagset_size+2] FloatTensor
            lens: [batch_size] LongTensor
        Return:
            partition_norm: [batch_size] LongTensor
        '''
        batch_size, max_length, _ = logits.size()
        alpha = torch.Tensor(batch_size, self.tagset_size).fill_(-10000.)
        alpha[:, self.START_TAG] = 0
        alpha = autograd.Variable(alpha)
        c_lens = lens.clone()

        logits_t = logits.transpose(1,0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(2).expand(batch_size, self.tagset_size, self.tagset_size)
            alpha_exp = alpha.unsqueeze(1).expand(batch_size, self.tagset_size, self.tagset_size)
            trans_exp = self.transitions.unsqueeze(0).expand(batch_size, self.tagset_size, self.tagset_size)
            mat = trans_exp + alpha_exp + logit_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            c_lens = c_lens - 1

        alpha = alpha + self.transitions[self.STOP_TAG].unsqueeze(0).expand_as(alpha)
        partition_norm = log_sum_exp(alpha, 1).squeeze(-1)

        return partition_norm
    
    def _transition_score(self, labels, lens):
        """
        Arguments:
            labels: [batch_size, max_length] LongTensor
            lens: [batch_size] LongTensor
        Return:
            score: [batch_size] LongTensor
        """
        batch_size, seq_len = labels.size()

        # pad labels with <start> and <stop> indices
        labels_ext = autograd.Variable(labels.data.new(batch_size, seq_len + 2))
        labels_ext[:, 0] = self.START_TAG
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=seq_len + 2).long()
        pad_stop = autograd.Variable(labels.data.new(1).fill_(self.STOP_TAG))
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 + (-1) * mask) * pad_stop + mask * labels_ext
        
        trn = self.transitions

        # obtain transition vector for each label in batch and timestep
        # (except the last ones)
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels_ext[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(lbl_r.size()[0], lbl_r.size()[1],trn.size(0))

        trn_row = torch.gather(trn_exp, 1, lbl_rexp)
        
        # obtain transition score from the transition vector for each label
        # in batch and timestep (except the first ones)
        lbl_lexp = labels_ext[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr.sum(1).squeeze(-1)

        return score

    def _emission_score(self, logits, labels, lens):
        """
        Arguments:
            logits: [batch_size, max_length, tagset_size+2] FloatTensor
            labels: [batch_size, max_length] LongTensor
            lens: [batch_size] LongTensor
        Return:
            score: [batch_size] LongTensor
        """
        labels_exp = labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores * mask
        score = scores.sum(1).squeeze(-1)

        return score
        
    def get_neg_log_likilihood_loss(self, logits, labels, lens):
        """
        Arguments:
            logits: [batch_size, max_length, tagset_size+2] FloatTensor
            labels: [batch_size, max_length] LongTensor
            lens: [batch_size] LongTensor
        Return:
            loss: LongTensor
        """
        partition_norm = self._forward_alg(logits, lens)
        transition_score = self._transition_score(labels, lens)
        emission_score = self._emission_score(logits, labels, lens)

        batch_size = logits.size()[0]
        score = partition_norm - emission_score - transition_score

        return  (1.0 / batch_size) * torch.sum(score)
    
    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, max_length, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        Return:
            scores: [batch_size] LongTensor
            path: [batch_size, max_length] LongTensor
        """
        batch_size, max_length, _ = logits.size()
        vit = logits.data.new(batch_size, self.tagset_size).fill_(-10000.)
        vit[:, self.START_TAG] = 0
        vit = autograd.Variable(vit)
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, self.tagset_size, self.tagset_size)
            trn_exp = self.transitions.unsqueeze(0).expand(batch_size, self.tagset_size, self.tagset_size)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2, keepdim=True)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transitions[ self.STOP_TAG ].unsqueeze(0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1, keepdim = True)
        idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]

        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths


    def forward(self, logits):
        '''
        Arguments:
            logits: [batch_size, max_length, tagset_size+2] FloatTensor
        Return:
            alpha: [batch_size, max_length, tagset_size+2] LongTensor
        '''
        batch_size, max_length, _ = logits.size()
        alpha_step = torch.Tensor(batch_size, self.tagset_size).fill_(-10000.)
        alpha_step[:, self.START_TAG] = 0
        alpha_step = autograd.Variable(alpha_step)
        alpha = autograd.Variable(torch.Tensor(batch_size, max_length, self.tagset_size))
        c_lens = lens.clone()

        logits_t = logits.transpose(1,0)
        for index, logit in enumerate(logits_t):
            logit_exp = logit.unsqueeze(2).expand(batch_size, self.tagset_size, self.tagset_size)
            alpha_exp = alpha_step.unsqueeze(1).expand(batch_size, self.tagset_size, self.tagset_size)
            trans_exp = self.transitions.unsqueeze(0).expand(batch_size, self.tagset_size, self.tagset_size)
            mat = trans_exp + alpha_exp + logit_exp
            alpha_step = log_sum_exp(mat, 2).squeeze(-1)
            alpha[:, index, :] = alpha_step.copy()

        return alpha
