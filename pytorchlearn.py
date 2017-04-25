from LstmCrfModel import BiLSTM_CRF
from collections import OrderedDict
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

tags = torch.LongTensor([0, 2, 3])
score = autograd.Variable(torch.Tensor([[1.,2.,3.,4.],[1.,2.,3.,4.],[1.,2.,3.,4.]]))
loss = autograd.Variable(torch.Tensor([0.]))

Q = nn.Sigmoid()
for tag, log_p in zip(tags, score):
    Pw = log_p[tag]
    log_p[tag] = log_p[tag] + 10000.
    maxPw = torch.max(log_p)
    log_p[tag] = log_p[tag] + 10000.
    print(Pw)
    #loss = loss + Q(Pw - maxPw)

