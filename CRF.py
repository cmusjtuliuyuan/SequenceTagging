import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
START_TAG = -2
STOP_TAG = -1


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec, dim=0):
    max, idx = torch.max(vec, dim, keepdim=True)
    max_exp = max.expand_as(vec)
    return max.squeeze(-1) + torch.log(torch.sum(torch.exp(vec - max_exp), dim))

def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().data[0]

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = autograd.Variable(ranges)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask

class CRF(nn.Module):

    def __init__(self, tagset_size):
        super(CRF, self).__init__()
        # We add 2 here, because of START_TAG and STOP_TAG
        self.tagset_size = tagset_size + 2
        
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

    def _forward_alg(self, logits, lens):
        '''
        Arguments:
            logits: [batch_size, max_length, tagset_size+2] FloatTensor
            lens: [batch_size] LongTensor
        Return:
            partition_norm: [batch_size] LongTensor
        borrow from: https://github.com/kaniblu/pytorch-bilstmcrf/blob/master/model.py
        '''
        batch_size, max_length, _ = logits.size()
        alpha = torch.Tensor(batch_size, self.tagset_size).fill_(-10000)
        alpha[:, START_TAG] = 0
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

        alpha = alpha + self.transitions[STOP_TAG].unsqueeze(0).expand_as(alpha)
        partition_norm = log_sum_exp(alpha, 1).squeeze(-1)

        return partition_norm
    
    def _transition_score(self, labels, lens):
        """
        Arguments:
             labels: [batch_size, seq_len] LongTensor
             lens: [batch_size] LongTensor
        """
        batch_size, seq_len = labels.size()

        # pad labels with <start> and <stop> indices
        labels_ext = autograd.Variable(labels.data.new(batch_size, seq_len + 2))
        labels_ext[:, 0] = START_TAG
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=seq_len + 2).long()
        pad_stop = autograd.Variable(labels.data.new(1).fill_(STOP_TAG))
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 + (-1) * mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transitions

        # obtain transition vector for each label in batch and timestep
        # (except the last ones)
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(lbl_r.size()[0], lbl_r.size()[1],trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        # obtain transition score from the transition vector for each label
        # in batch and timestep (except the first ones)
        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr.sum(1).squeeze(-1)

        return score

    def get_neg_log_likilihood_loss(self, logits, labels, lens):
        # nonegative log likelihood
        partition_norm = self._forward_alg(logits, lens)
        transition_score = self._transition_score(labels, lens)
        return partition_norm - transitions_score

    '''
    def _viterbi_decode(self, feats):
        backpointers = []
        
        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size+2).fill_(-10000.)
        init_vvars[0][ START_TAG ] = 0
        
        # forward_var at step i holds the viterbi variables for step i-1 
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = [] # holds the backpointers for this step
            viterbivars_t = [] # holds the viterbi variables for this step
            
            for next_tag in xrange(self.tagset_size+2):
                # next_tag_var[i] holds the viterbi variable for tag i at the previous step,
                # plus the score of transitioning from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[ STOP_TAG ]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()

        best_path.reverse()
        return path_score, best_path


    def forward(self, feats):
    	score, _ = self._viterbi_decode(feats)
    	return score
    

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable( torch.Tensor([0]) )

        tags = tags.data.numpy()
        tags = np.concatenate(([START_TAG], tags), axis=0)
        
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
        score = score + self.transitions[STOP_TAG, tags[-1]]
        return score
    '''