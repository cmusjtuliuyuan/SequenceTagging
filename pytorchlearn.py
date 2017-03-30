from LstmCrfModel import BiLSTM_CRF
from collections import OrderedDict
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


def prepare_sequence(seq, to_ix):
    idxs = map(lambda w: to_ix[w], seq)
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

training_data = [ (
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
) ]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = { "B": 0, "I": 1, "O": 2 }

Model_parameters = OrderedDict()
Model_parameters['vocab_size'] = 18
Model_parameters['embedding_dim'] = 20
Model_parameters['hidden_dim'] = 10
Model_parameters['tagset_size'] = 3

model = BiLSTM_CRF(Model_parameters)
sentence_in = prepare_sequence(training_data[0][0], word_to_ix)
targets_in = torch.LongTensor([ tag_to_ix[t] for t in training_data[0][1] ])


print(model.neg_log_likelihood(sentence_in, targets_in))
print(model(sentence_in))