import os
import re
import codecs
from utils import create_dico, create_mapping, zero_digits
from utils import read_pre_training
import numpy as np
import string 

FEATURE_DIM = {
    'input_caps': 4,
    'input_letter_digits': 4,
    'input_apostrophe_ends': 2,
    'input_punctuations': 2,
}

def load_sentences(path, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def check_tag_chunking(sentences):
    """
    Check the input format is chunking or not 
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        for j, tag in enumerate(tags):
            if tag == 'O':
                continue
            split = tag.split('-')
            #if len(split) != 2 or split[0] not in ['I', 'B'] \
            #            or split[1] not in ['NP', 'VP', 'PP', 'SBAR', 'ADVP','ADJP']:
            #    print(split)
            #    raise Exception('Unknown tagging scheme!')



def word_mapping(sentences, lower,vocabulary_size, pre_train = None):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    word_to_id, id_to_word = create_mapping(dico, vocabulary_size)
    print ("Found %i unique words (%i in total)" % 
        (len(dico), sum(len(x) for x in words))
    )

    if pre_train:
        emb_dictionary = read_pre_training(pre_train)
        for word in dico.iterkeys():
        	  if word not in emb_dictionary:
        	  	  dico[word]=0
        	  	  
    dico['<UNK>'] = 10000000
    # ADD <EMPTY> for batch
    dico['<EMPTY>'] = 10000001
    word_to_id, id_to_word = create_mapping(dico, vocabulary_size)

    return dico, word_to_id, id_to_word


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    # ADD <EMPTY> for batch
    dico['<EMPTY>'] = 10000001
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % (len(dico)))
    return dico, tag_to_id, id_to_tag

def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3

def letter_digit_feature(s):
    if re.search('[a-zA-Z]',s) and re.search('[0-9]',s):
        return 0
    elif re.search('[a-zA-Z]',s):
        return 1
    elif re.search('[0-9]',s):
        return 2
    else:
        return 3

def apostrophe_end_feature(s):
    if len(s)>1 and s[-2:] == "'s":
        return 0
    else:
        return 1


def punctuation_feature(s):
    if re.search('['+string.punctuation+']', 'a'):
        return 0
    else: 
        return 1


def prepare_dataset(sentences, word_to_id, tag_to_id, lower=False, supervised = True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        letter_digits = [letter_digit_feature(w) for w in str_words]
        apostrophe_ends = [apostrophe_end_feature(w) for w in str_words]
        punctuations = [punctuation_feature(w) for w in str_words]
        data.append({
            'str_words': str_words,
            'words': words,
            'caps': caps,
            'letter_digits': letter_digits,
            'apostrophe_ends': apostrophe_ends,
            'punctuations': punctuations,
        })
        if supervised:
            pos = [w[1] for w in s]
            tags = [tag_to_id[w[-1]] for w in s]
            data[-1]['pos']=pos;
            data[-1]['tags']=tags;
    return data

def prepare_dictionaries(parameters):
    lower = parameters['lower']
    zeros = parameters['zeros']
    train_path = parameters['train']
    dev_path = parameters['development']
    vocabulary_size = parameters['vocab_size']

    # Load sentences
    train_sentences = load_sentences(train_path, zeros)
    # Use selected tagging scheme
    check_tag_chunking(train_sentences)

    if parameters['pre_emb']:
        dev_sentences = load_sentences(dev_path,  zeros)
        sentences = train_sentences + dev_sentences
        dico_words, word_to_id, id_to_word = word_mapping(sentences, 
                                   lower,vocabulary_size, parameters['pre_emb'])
    else:
        dico_words, word_to_id, id_to_word = word_mapping(train_sentences, 
                                    lower,vocabulary_size, parameters['pre_emb'])
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

    dictionaries = {
        'word_to_id': word_to_id,
        'id_to_word': id_to_word,
        'tag_to_id': tag_to_id,
        'id_to_tag': id_to_tag,
    }

    return dictionaries

def load_dataset(parameters, path, dictionaries, supervised = True):
    # Data parameters
    lower = parameters['lower']
    zeros = parameters['zeros']

    # Load sentences
    sentences = load_sentences(path, zeros)
    dataset = prepare_dataset(
        sentences, dictionaries['word_to_id'], dictionaries['tag_to_id'], lower, supervised
    )
    print("%i sentences in %s ."%(len(dataset), path))
    return dataset

def get_word_embedding_matrix(dictionary, pre_train, embedding_dim):
    emb_dictionary = read_pre_training(pre_train)
    dic_size = len(dictionary)
    initial_matrix = np.random.random(size=(dic_size, embedding_dim))
    for word, idx in dictionary.iteritems(): 
        if word != '<UNK>':
            initial_matrix[idx] = emb_dictionary[word]
    return initial_matrix
