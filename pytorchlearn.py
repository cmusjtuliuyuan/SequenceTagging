from LstmCrfModel import BiLSTM_CRF
from collections import OrderedDict
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


# Helper functions to make the code more readable.
def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp1(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp(vec):
    return torch.log(torch.sum(torch.exp(vec)))

class TEST():

    def __init__(self):
        self.tagset_size = 2
        self.transitions = nn.Parameter(torch.randn(self.tagset_size+2, self.tagset_size+2))
        feats = autograd.Variable(torch.Tensor([[1.,1.,1.,1.], [2.,2.,2.,2.]]))
        print(self._backward_alg(feats))

    def _backward_alg(self, feats):
        # Do the backward algorithm
        # ADD 2 here because of START_TAG and STOP_TAG 
        init_betas = torch.Tensor(1, self.tagset_size+2).fill_(0)
        
        # Wrap in a variable so that we will get automatic backprop
        # This is beta_{T+1} vector 
        backward_var = autograd.Variable(init_betas)
        
        # Iterate through the sentence
        beta = []

        # First calculate beta_{T}, because we do not have Emition_matrix{, T+1}
        # so we need to calculate it seperately
        betas_t = [] 
        next_tag_var = autograd.Variable(torch.Tensor(1, self.tagset_size+2).fill_(0.))

        for next_tag in xrange(self.tagset_size+2):
            # We add transition score in this way because 
            #self.transition[:, next_tag] will not be contiguous, so we cannot use view function
            for i, trans_val in enumerate(self.transitions[:,next_tag]):
                next_tag_var[0,i] = backward_var[0,i]+trans_val
            betas_t.append(log_sum_exp(next_tag_var))
        backward_var = torch.cat(betas_t).view(1, -1)
        beta.append(backward_var)


        # Second we can begin the loop
        # became with Emition_matrix{, T}, beta_{T} to calulate beta_{T-1}
        for feat in feats[1::-1]:
            betas_t = []
            #alphas_t = [] # The forward variables at this timestep
            for next_tag in xrange(self.tagset_size+2):

                emit_score = feat.view(1, -1)
                
                next_tag_var = backward_var + emit_score

                for i, trans_val in enumerate(self.transitions[:,next_tag]):
                    next_tag_var[0,i] += trans_val

                betas_t.append(log_sum_exp(next_tag_var))
            backward_var = torch.cat(betas_t).view(1, -1)
            beta.append(backward_var)

        log_beta = torch.cat(beta[::-1] , 0)
        return log_beta

test = TEST()