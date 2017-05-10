import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loader import CAP_DIM
START_TAG = -2
STOP_TAG = -1
DROP_OUT = 0.5



# Helper functions to make the code more readable.
def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    

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
        
        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size+2, self.tagset_size+2))
        
        self.hidden = self.init_hidden()

    def init_word_embedding(self, init_matrix):
        self.word_embeds.weight=nn.Parameter(torch.FloatTensor(init_matrix))


    def init_hidden(self):
        return ( autograd.Variable( torch.randn(2, 1, self.hidden_dim)),
                 autograd.Variable( torch.randn(2, 1, self.hidden_dim)) )


    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        # ADD 2 here because of START_TAG and STOP_TAG
        init_alphas = torch.Tensor(1, self.tagset_size+2).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][ START_TAG ] = 0.
        
        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)
        
        # Iterate through the sentence
        alpha = []
        for feat in feats:
            alphas_t = [] # The forward variables at this timestep
            for next_tag in xrange(self.tagset_size+2):
                # broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size+2)
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the edge (i -> next_tag)
                # before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
            alpha.append(forward_var)

        terminal_var = forward_var + self.transitions[ STOP_TAG ]
        log_partition_Z = log_sum_exp(terminal_var)
        log_alpha = torch.cat(alpha , 0)
        return log_partition_Z, log_alpha


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
        # slice step has to be greater than 0!!! so urgely in Pytorch
        for j in range(len(feats), 1, -1):
            feat = feats[j-1]
            betas_t = []
            #alphas_t = [] # The forward variables at this timestep
            for next_tag in xrange(self.tagset_size+2):

                emit_score = feat.view(1, -1)
                
                next_tag_var = backward_var + emit_score
                # 
                for i, trans_val in enumerate(self.transitions[:,next_tag]):
                    next_tag_var[0,i] = next_tag_var[0,i] + trans_val

                betas_t.append(log_sum_exp(next_tag_var))
            backward_var = torch.cat(betas_t).view(1, -1)
            beta.append(backward_var)

        log_beta = torch.cat(beta[::-1] , 0)
        return log_beta


    def _marginal_decode(self, feats):
        # Use forward backward algorithm to calculate the marginal distribution
        # Decode according to the marginal distribution.
        _, log_alpha = self._forward_alg(feats)
        log_beta = self._backward_alg(feats)
        score = log_alpha+log_beta
        _, tags = torch.max(score, 1)
        tags = tags.view(-1).data.tolist()
        return score, tags


    def _get_lstm_features(self, dropout, **sentence):
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


    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable( torch.Tensor([0]) )

        tags = tags.data.numpy()
        tags = np.concatenate(([START_TAG], tags), axis=0)
        
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
        score = score + self.transitions[STOP_TAG, tags[-1]]
        return score


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

    def get_loss(self, tags, **sentence):
        if self.loss_function == 'likelihood':
            return self.get_neg_log_likilihood_loss(tags, **sentence)
        elif self.loss_function == 'labelwise':
            return self.get_labelwise_loss(tags, **sentence)
        else:
            print("ERROR: The parameter of loss function is wrong")


    def get_neg_log_likilihood_loss(self, tags, **sentence):
        input_words = sentence['input_words']

        if self.lower:
            input_caps = sentence['input_caps']
            feats = self._get_lstm_features(dropout = True, input_words = input_words,
                                  input_caps = input_caps)
        else:
            feats = self._get_lstm_features(dropout = True, input_words = input_words)

        # nonegative log likelihood
        #feats = self._get_lstm_features(sentence)
        forward_score, _ = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def get_labelwise_loss(self, tags, **sentence):
        # Get the emission scores from the BiLSTM
        input_words = sentence['input_words']

        if self.lower:
            input_caps = sentence['input_caps']
            feats = self._get_lstm_features(dropout = True, input_words = input_words,
                                  input_caps = input_caps)
        else:
            feats = self._get_lstm_features(dropout = True, input_words = input_words)
        #lstm_feats = self._get_lstm_features(sentence)
        
        # Get the marginal distribution
        score, _ = self._marginal_decode(feats)
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
        input_words = sentence['input_words']

        if self.lower:
            input_caps = sentence['input_caps']
            feats = self._get_lstm_features(dropout = False, input_words = input_words,
                                  input_caps = input_caps)
        else:
            feats = self._get_lstm_features(dropout = False, input_words = input_words)
        #lstm_feats = self._get_lstm_features(sentence)
        
        # Find the best path, given the features.
        if self.decode_method == 'marginal':
            score, tag_seq = self._marginal_decode(feats)
        elif self.decode_method == 'viterbi':
            score, tag_seq = self._viterbi_decode(feats)
        else:
            print("Error wrong decode method")

        return score, tag_seq


    def get_tags(self, **sentence):
        input_words = sentence['input_words']

        if self.lower:
            input_caps = sentence['input_caps']
            _, tag_seq = self.forward(input_words = input_words,
                                  input_caps = input_caps)
        else:
            _, tag_seq = self.forward(input_words = input_words)

        #score, tag_seq = self.forward(sentence)
        return np.asarray(tag_seq).reshape((-1,))
