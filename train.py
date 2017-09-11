import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loader import load_dataset
import codecs
import os


BATCH_SIZE = 32
LEARNING_RATE = 0.1
EVALUATE_EVERY = 3
NUM_EPOCH = 30

def adjust_learning_rate(optimizer, lr, epoch):
    true_lr = lr * (0.5 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = true_lr


def train(model, Parse_parameters, opts, dictionaries):
    train_data = load_dataset(Parse_parameters, opts.train, dictionaries)
    dev_data = load_dataset(Parse_parameters, opts.dev, dictionaries)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in xrange(1, NUM_EPOCH+1): 
        print("Trian epoch: %d"%(epoch))

        adjust_learning_rate(optimizer, LEARNING_RATE , epoch)
        train_epoch(model, train_data, opts, optimizer)

        if epoch % EVALUATE_EVERY == 0:
            evaluate(model, dev_data, dictionaries)


def train_epoch(model, train_data, opts, optimizer):

    def train_batch(model, sentences, opts, optimizer):
        model.zero_grad()
        loss = model.get_loss(sentences)
        loss.backward()
        #print loss.data
        nn.utils.clip_grad_norm(model.parameters(), opts.clip)
        optimizer.step()

    sentences = []
    for i, index in enumerate(np.random.permutation(len(train_data))):
        # Prepare batch dataset
        sentences.append(train_data[index])
        if len(sentences) == BATCH_SIZE:
            # Train the model
            train_batch(model, sentences, opts, optimizer)
            # Clear old batch
            sentences = []

    if len(sentences) != 0:
        train_batch(model, sentences, opts, optimizer)


def evaluate(model, dev_data, dictionaries):

    def evaluate_batch(model, sentences, dictionaries, file):
        preds = model.get_tags(sentences)

        for sentence, pred in zip(sentences, preds):

            # get predict tags
            predict_tags = [dictionaries['id_to_tag'][tag] if (tag in dictionaries['id_to_tag']) else 'START_STOP' for tag in pred]

            # get true tags
            true_tags = [dictionaries['id_to_tag'][tag] for tag in sentence['tags']]

            # write words pos true_tag predict_tag into a file
            for word, pos, true_tag, predict_tag in zip(sentence['str_words'], 
                                                    sentence['pos'],
                                                    true_tags, predict_tags):
                file.write('%s %s %s %s\n' % (word, pos ,true_tag, predict_tag))
            file.write('\n')

    """
    Evaluate current model using CoNLL script.
    """
    output_path = 'tmp/evaluate.txt'
    scores_path = 'tmp/score.txt'
    eval_script = './tmp/conlleval'
    with codecs.open(output_path, 'w', 'utf8') as f:
        sentences = []

        for index in xrange(len(dev_data)):
            sentences.append(dev_data[index])

            if len(sentences) == BATCH_SIZE:
                evaluate_batch(model, sentences, dictionaries, f)
                sentences = []
        if len(sentences) != 0:
            evaluate_batch(model, sentences, dictionaries, f)

    os.system("%s < %s > %s" % (eval_script, output_path, scores_path))
    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
    result={
       'accuracy' : float(eval_lines[1].strip().split()[1][:-2]),
       'precision': float(eval_lines[1].strip().split()[3][:-2]),
       'recall': float(eval_lines[1].strip().split()[5][:-2]),
       'FB1': float(eval_lines[1].strip().split()[7])
    }
    print(eval_lines[1])
    return result
