import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loader import load_dataset
from utils import evaluate

BATCH_SIZE = 1

def train(model, Parse_parameters, opts, dictionaries):
    train_data = load_dataset(Parse_parameters, opts.train, dictionaries)
    dev_data = load_dataset(Parse_parameters, opts.dev, dictionaries)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Number of Epoch, It needs 10 epoches to converge
    n_epochs = 1
    #eval_epoch(model, dev_data, dictionaries, opts)
    for epoch in xrange(n_epochs): 
        print("Trian epoch: %d"%(epoch))
        train_epoch(model, train_data, opts, optimizer)
    eval_epoch(model, dev_data, dictionaries, opts)

def train_epoch(model, train_data, opts, optimizer):
    sentences = []
    for i, index in enumerate(np.random.permutation(len(train_data))):
        # Prepare batch dataset
        sentences.append(train_data[index])
        if i!=0 and i%BATCH_SIZE==BATCH_SIZE-1:
            # Train the model
            
            model.zero_grad()
            loss = model.get_loss(sentences)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), opts.clip)
            optimizer.step()
            #print loss.data
            # Clear old batch
            sentences = []
    print model.get_tags([train_data[1]])


def eval_epoch(model, dev_data, dictionaries, opts):
    eval_result = evaluate(model, dev_data, dictionaries, opts.lower)
    return eval_result