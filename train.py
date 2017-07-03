import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loader import load_dataset
from utils import evaluate

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
    model.train()
    for i, index in enumerate(np.random.permutation(100)):#len(train_data))):
            # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
        # before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into Variables
        # of word indices.
        input_words = autograd.Variable(torch.LongTensor(train_data[index]['words']))
        targets = autograd.Variable(torch.LongTensor(train_data[index]['tags']))

        # Step 3. Run our forward pass. We combine this step with get_loss function
        #tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by calling
        # first check whether we use lower this parameter
        if opts.lower == 1:
            input_caps = torch.LongTensor(train_data[index]['caps'])
            input_letter_digits = torch.LongTensor(train_data[index]['letter_digits'])
            input_apostrophe_ends = torch.LongTensor(train_data[index]['apostrophe_ends'])
            input_punctuations = torch.LongTensor(train_data[index]['punctuations'])
            loss = model.get_loss(targets,
                                  input_words = input_words,
                                  input_caps = input_caps,
                                  input_letter_digits = input_letter_digits,
                                  input_apostrophe_ends = input_apostrophe_ends,
                                  input_punctuations = input_punctuations )
        else:
            loss = model.get_loss(targets, input_words = input_words)

        #epoch_costs.append(loss.data.numpy())
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), opts.clip)
        optimizer.step()

def eval_epoch(model, dev_data, dictionaries, opts):
    eval_result = evaluate(model, dev_data, dictionaries, opts.lower)
    return eval_result