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
    if tag == 0:
        not_tag = log_p[1:]
    elif tag == len(log_p) - 1:
        not_tag = log_p[:tag]
    else:
        not_tag = torch.cat((log_p[:tag], log_p[tag+1:]))
    maxPw = torch.max(not_tag)
    loss = loss + Q(Pw - maxPw)
print(loss)
