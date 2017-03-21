import os
import re
import numpy as np
import torch
import torch.autograd as autograd
import codecs


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

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

def evaluate(model, sentences, dictionaries):
    """
    Evaluate current model using CoNLL script.
    """
    output_path = 'tmp/evaluate.txt'
    scores_path = 'tmp/score.txt'
    eval_script = './tmp/conlleval'
    with codecs.open(output_path, 'w', 'utf8') as f:
       for index in xrange(len(sentences)):
            #input sentence
            sentence_in = autograd.Variable(torch.LongTensor(sentences[index]['words']))

            #calculate the tag score
            tag_scores = model(sentence_in).data.numpy()

            # get predict tags
            tags = np.argmax(tag_scores, axis=1)
            predict_tags = [dictionaries['id_to_tag'][tag] for tag in tags]

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
    