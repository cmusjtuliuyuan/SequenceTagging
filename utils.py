import os
import re
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import codecs
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import json


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico, vocabulary_size=2000):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), 
            key=lambda x: (-x[1], x[0]))[:vocabulary_size]
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def read_pre_training(emb_path):
    """
    Read pre-train word embeding
    The detail of this dataset can be found in the following link
    https://nlp.stanford.edu/projects/glove/ 
    """
    print('Preparing pre-train dictionary')
    emb_dictionary={}
    for line in codecs.open(emb_path, 'r', 'utf-8'):
        temp = line.split()
        emb_dictionary[temp[0]] = np.asarray(temp[1:], dtype= np.float16)
    return emb_dictionary


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

def plot_result(accuracys, precisions, recalls, FB1s):
    plt.figure()
    plt.plot(accuracys,"g-",label="accuracy")
    plt.plot(precisions,"r-.",label="precision")
    plt.plot(recalls,"m-.",label="recalls")
    plt.plot(FB1s,"k-.",label="FB1s")

    plt.xlabel("epoches")
    plt.ylabel("%")
    plt.title("CONLL2000 dataset")

    plt.grid(True)
    plt.legend()
    plt.show()

def save_model_dictionaries(path, model, dictionaries, opts):
    """
    We need to save the mappings if we want to use the model later.
    """
    print("Model is saved in:"+path)
    with open(path+'/dictionaries.dic', 'wb') as f:
        cPickle.dump(dictionaries, f)
    torch.save(model.state_dict(), path+'/model.mdl')
    with open(path+'/parameters.json', 'w') as outfile:
    	json.dump(vars(opts), outfile, sort_keys = True, indent = 4)

def load_parameters(path, opts):
    param_file = os.path.join(path, 'parameters.json')
    with open(param_file, 'r') as file:
        params = json.load(file)
        # Read network architecture parameters from previously saved
        # parameter file.
        opts.clip = params['clip']
        opts.decode_method = params['decode_method']
        opts.embedding_dim = params['embedding_dim']
        opts.freeze = params['freeze']
        opts.hidden_dim = params['hidden_dim']
        opts.loss_function = params['loss_function']
        opts.lower = params['lower']
        opts.vocab_size = params['vocab_size']
        opts.zeros = params['zeros']
    return opts

def sentences2padded(sentences, keyword, replace = 0):
    # Form Batch_Size * Length
    max_length = max([len(sentence[keyword]) for sentence in sentences])
    def pad_seq(seq, max_length):
        padded_seq = seq + [replace for i in range(max_length - len(seq))]
        return padded_seq
    padded =[pad_seq(sentence[keyword], max_length) for sentence in sentences]
    return padded

def get_lens(sentences, keyword):
    return [len(sentence[keyword]) for sentence in sentences]

# Return mask matrix
def sequence_mask(lens, max_len=None, cuda = False):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().data[0]

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = autograd.Variable(ranges)

    if cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask

def char2padded(sentences, replace = 0):
    def pad_seq_forward(seq, max_char_length):
        padded_seq = seq + [replace for i in range(max_char_length - len(seq))]
        return padded_seq
    def pad_seq_backward(seq, max_char_length):
        padded_seq = seq[::-1] + [replace for i in range(max_char_length - len(seq))]
        return padded_seq

    forward_chars=[]
    backward_chars=[]
    lens_chars=[]

    max_length = max([len(sentence['words']) for sentence in sentences])
    for i in range(max_length):
        max_char_length = max([len(sentence['chars'][i]) 
                        if len(sentence['chars'])>i else 0 
                        for sentence in sentences])
        forward_padded =[pad_seq_forward(sentence['chars'][i], max_char_length)
                        if len(sentence['chars'])>i else pad_seq_forward([], max_char_length)
                        for sentence in sentences]

        backward_padded =[pad_seq_backward(sentence['chars'][i], max_char_length)
                        if len(sentence['chars'])>i else pad_seq_backward([], max_char_length)
                        for sentence in sentences]

        length = [len(sentence['chars'][i]) if len(sentence['chars'])>i else 0 
                        for sentence in sentences]

        forward_chars.append(forward_padded)
        backward_chars.append(backward_padded)
        lens_chars.append(length)
    return forward_chars, backward_chars, lens_chars
