import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loader import load_dataset
import codecs
import os
from Autoencoder import grads
import re

BATCH_SIZE = 32
LEARNING_RATE = 0.1
NUM_EPOCH = 1
US_FACTOR = 2
SUPERVISED = True
UNSUPERVISED = False

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_supervised_then_unsupervised(model, optimizer_s, optimizer_us,
        train_data, dev_data, dictionaries, epoch, files, opts, Parse_parameters):
    
    def train_until_overfit(begin, end):
        overfit = False
        FB1array = []
        while not overfit:
            train_epoch(model, train_data, opts, optimizer_s, SUPERVISED)
            result = evaluate(model, dev_data, dictionaries)

            FB1array.append(result['FB1'])

            # save the best model:
            if result['FB1'] == max(FB1array):
                torch.save(model.state_dict(), TEMP_PATH)

            # check overfit:
            if len(FB1array)>begin:
                if FB1array[-1]<FB1array[-2] and FB1array[-2]<FB1array[-3]:
                    overfit = True
        
            if len(FB1array)>end:
                overfit = True

    # supervised train, early stop and adjust learning rate
    TEMP_PATH = 'models/tmp_model.mdl'
    adjust_learning_rate(optimizer_s, LEARNING_RATE)
    train_until_overfit(5,15)
    model.load_state_dict(torch.load(TEMP_PATH))
    adjust_learning_rate(optimizer_s, LEARNING_RATE/10)
    train_until_overfit(5,10)
    model.load_state_dict(torch.load(TEMP_PATH))

    # unsupervised train
    for i in range(US_FACTOR):
        unlabel_data_file_name = files[(US_FACTOR*epoch + i - 1)%len(files)]
        unlabel_data = load_dataset(Parse_parameters,
            'data/wiki100/'+unlabel_data_file_name, dictionaries, UNSUPERVISED)
        train_epoch(model, unlabel_data, opts, optimizer_us, UNSUPERVISED)

def train(model, Parse_parameters, opts, dictionaries):
    # Prepare unsupervised dataset:
    path = 'data/wiki100'
    TEMP_PATH = 'models/tmp_model.mdl'
    files = os.listdir(path)
    files = [ x for x in files if 'wiki100' in x]
    files.sort()
    train_data = load_dataset(Parse_parameters, opts.train, dictionaries)
    dev_data = load_dataset(Parse_parameters, opts.dev, dictionaries)
    optimizer_s = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    optimizer_us = optim.SGD(model.embeds_parameters+\
                            list(model.decoder.parameters()), lr=LEARNING_RATE)
    if opts.load:
        epoch_begin = int(re.search(r'\d+', opts.load).group())+1
        model.load_state_dict(torch.load(opts.load))
	print 'load:', opts.load
    else:
        epoch_begin = 1
    print 'Start epoch from:', epoch_begin

    for epoch in xrange(epoch_begin, epoch_begin+NUM_EPOCH):
        print("Train epoch: %d"%(epoch))
        train_supervised_then_unsupervised(model, optimizer_s, optimizer_us,
                train_data, dev_data, dictionaries, epoch, files, opts, Parse_parameters)
        torch.save(model.state_dict(), 'models/model%d.mdl'%(epoch,))


def train_epoch(model, train_data, opts, optimizer, supervised = True):

    def train_batch(model, sentences, opts, optimizer, supervised = True):
        sentences.sort(key = lambda sentence: -len(sentence['words']))

        model.zero_grad()
        if supervised:
            loss = model.get_loss_supervised(sentences)
        else:
            loss = model.get_loss_unsupervised(sentences)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), opts.clip)
        optimizer.step()

    model.train()
    sentences = []
    # increase the batchsize in unsupervised training to speed up
    if supervised:
        batchsize = BATCH_SIZE
    else:
        batchsize = 4*BATCH_SIZE
    for i, index in enumerate(np.random.permutation(len(train_data))):
        # Prepare batch dataset
        sentences.append(train_data[index])
        if len(sentences) == batchsize:
            # Train the model
            train_batch(model, sentences, opts, optimizer, supervised)
            # Clear old batch
            sentences = []

    if len(sentences) != 0:
        train_batch(model, sentences, opts, optimizer, supervised)


def evaluate(model, dev_data, dictionaries):

    def evaluate_batch(model, sentences, dictionaries, file):
        # First sort sentences
        ordered_sentences = sorted(sentences, key = lambda sentence: -len(sentence['words']))
        index = sorted(range(len(sentences)), key=lambda k: -len(sentences[k]['words']))
        # Then predict
        ordered_preds = model.get_tags(ordered_sentences)
        # Finaly recover the sentences
        preds = ordered_preds[:]
        for o_p, i in zip(ordered_preds, index):
            preds[i] = o_p
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
    model.eval()
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
