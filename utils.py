import os
import re
import numpy as np
import torch
import torch.autograd as autograd
import codecs
import numpy as np
import matplotlib.pyplot as plt
import cPickle


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

def evaluate(model, sentences, dictionaries, lower):
    """
    Evaluate current model using CoNLL script.
    """
    output_path = 'tmp/evaluate.txt'
    scores_path = 'tmp/score.txt'
    eval_script = './tmp/conlleval'
    with codecs.open(output_path, 'w', 'utf8') as f:
       for index in xrange(len(sentences)):
            #input sentence
            input_words = autograd.Variable(torch.LongTensor(sentences[index]['words']))
            
            #calculate the tag score
            if lower == 1:
                input_caps = torch.LongTensor(sentences[index]['caps'])
                input_letter_digits = torch.LongTensor(sentences[index]['letter_digits'])
                input_apostrophe_ends = torch.LongTensor(sentences[index]['apostrophe_ends'])
                input_punctuations = torch.LongTensor(sentences[index]['punctuations'])
                tags = model.get_tags(input_words = input_words,
                                      input_caps = input_caps,
                                      input_letter_digits = input_letter_digits,
                                      input_apostrophe_ends = input_apostrophe_ends,
                                      input_punctuations = input_punctuations )
            else:
                tags = model.get_tags(input_words = input_words)

            #tags = model.get_tags(sentence_in)
            # get predict tags
            predict_tags = [dictionaries['id_to_tag'][tag] if (tag in dictionaries['id_to_tag']) else 'START_STOP' for tag in tags]

            # get true tags
            true_tags = [dictionaries['id_to_tag'][tag] for tag in sentences[index]['tags']]

            # write words pos true_tag predict_tag into a file
            for word, pos, true_tag, predict_tag in zip(sentences[index]['str_words'], 
                                                        sentences[index]['pos'],
                                                        true_tags, predict_tags):
                f.write('%s %s %s %s\n' % (word, pos ,true_tag, predict_tag))
            f.write('\n')

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

def save_model_dictionaries(path, model, dictionaries):
    """
    We need to save the mappings if we want to use the model later.
    """
    print("Model is saved in:"+path)
    with open(path+'/dictionaries.dic', 'wb') as f:
        cPickle.dump(dictionaries, f)
    torch.save(model.state_dict(), path+'/model.mdl')