import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loader import load_dataset
from utils import evaluate

BATCH_SIZE = 32
LEARNING_RATE = 0.1
EVALUATE_EVERY = 3
NUM_EPOCH = 1

def adjust_learning_rate(optimizer, lr, epoch):
    true_lr = lr * (0.8 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = true_lr


def train(model, Parse_parameters, opts, dictionaries):
    train_data = load_dataset(Parse_parameters, opts.train, dictionaries)
    dev_data = load_dataset(Parse_parameters, opts.dev, dictionaries)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # Number of Epoch
    n_epochs = NUM_EPOCH
    #eval_epoch(model, dev_data, dictionaries, opts)
    for epoch in xrange(n_epochs): 
        print("Trian epoch: %d"%(epoch))

        adjust_learning_rate(optimizer, LEARNING_RATE , epoch)
        train_epoch(model, train_data, opts, optimizer)

        if epoch != 0 and (epoch+1)%EVALUATE_EVERY == 0:
            eval_epoch(model, dev_data, dictionaries, opts)


def train_epoch(model, train_data, opts, optimizer):
    sentences = []
    for i, index in enumerate(np.random.permutation(32)):
        # Prepare batch dataset
        sentences.append(train_data[index])
        if len(sentences) == BATCH_SIZE:
            '''
            # Train the model
            model.zero_grad()
            loss = model.get_loss(sentences)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), opts.clip)
            optimizer.step()
            '''
            model.forward(sentences)
            # Clear old batch
            sentences = []


def eval_epoch(model, dev_data, dictionaries, opts):
    eval_result = evaluate(model, dev_data, dictionaries, opts.lower)
    return eval_result