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
NUM_EPOCH = 100
SUPERVISED = True
UNSUPERVISED = False

def adjust_learning_rate(optimizer, lr, epoch):
    true_lr = lr * (0.9 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = true_lr

def train(model, Parse_parameters, opts, dictionaries):
    # Prepare unsupervised dataset:
    path = 'data/wiki'
    files = os.listdir(path)
    # filter .DS_Store ...
    files = [ x for x in files if not '.DS_Store' in x]

    train_data = load_dataset(Parse_parameters, opts.train, dictionaries)
    dev_data = load_dataset(Parse_parameters, opts.dev, dictionaries)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    '''
    for epoch in xrange(1, NUM_EPOCH+1): 
        print("Trian epoch: %d"%(epoch))

        adjust_learning_rate(optimizer, LEARNING_RATE , epoch)

        # supervised train
        train_epoch(model, train_data, opts, optimizer, SUPERVISED)
        evaluate(model, dev_data, dictionaries)

        # unsupervised train
        unlabel_data_file_name = files[(epoch-1)%len(files)]
        unlabel_data = load_dataset(Parse_parameters, 
                'data/wiki/'+unlabel_data_file_name, dictionaries, UNSUPERVISED)
        train_epoch(model, unlabel_data, opts, optimizer, UNSUPERVISED)
        evaluate(model, dev_data, dictionaries)

        #if epoch % EVALUATE_EVERY == 0:
        #    evaluate(model, dev_data, dictionaries)
    '''
    for epoch in xrange(1, 30+1):
        print("Trian epoch: %d"%(epoch))
        # unsupervised train
        unlabel_data_file_name = files[(epoch-1)%len(files)]
        unlabel_data = load_dataset(Parse_parameters,
                'data/wiki/'+unlabel_data_file_name, dictionaries, UNSUPERVISED)
        train_epoch(model, unlabel_data, opts, optimizer, UNSUPERVISED)

    for epoch in xrange(1, 20+1):
        print("Trian epoch: %d"%(epoch))
        train_epoch(model, train_data, opts, optimizer, SUPERVISED)
        evaluate(model, dev_data, dictionaries)


def train_epoch(model, train_data, opts, optimizer, supervised = True):

    def train_batch(model, sentences, opts, optimizer, supervised = True):
        sentences.sort(key = lambda sentence: -len(sentence['words']))

        model.zero_grad()
        if supervised:
            loss, flag = model.get_loss_supervised(sentences)
        else:
            loss, flag = model.get_loss_unsupervised(sentences)
        if flag:
            # flag is to check whether the max_length < 80 or not
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
