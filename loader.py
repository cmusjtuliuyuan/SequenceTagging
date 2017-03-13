import os
import re
import codecs
from utils import create_dico, create_mapping, zero_digits
import cPickle

def load_sentences(path, lower, zeros):
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
            if len(split) != 2 or split[0] not in ['I', 'B'] \
                        or split[1] not in ['NP', 'VP', 'PP', 'SBAR', 'ADVP','ADJP']:
                print(split)
                raise Exception('Unknown tagging scheme!')



def word_mapping(sentences, lower,vocabulary_size):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico, vocabulary_size)
    print ("Found %i unique words (%i in total)" % 
        (len(dico), sum(len(x) for x in words))
    )
    return dico, word_to_id, id_to_word


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % (len(dico)))
    return dico, tag_to_id, id_to_tag


def save_mappings(id_to_word, id_to_tag, mappings_path):
        """
        We need to save the mappings if we want to use the model later.
        """
        with open(mappings_path, 'wb') as f:
            mappings = {
                'id_to_word': id_to_word,
                'id_to_tag': id_to_tag,
            }
            cPickle.dump(mappings, f)


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


def prepare_dataset(sentences, word_to_id, tag_to_id, lower=False):
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
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'caps': caps,
            'tags': tags,
        })
    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % (ext_emb_path))
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def load_train_step_datasets(parameters):
    # Data parameters
    lower = parameters['lower']
    zeros = parameters['zeros']
    train_path = parameters['train']
    vocabulary_size = parameters['vocab_size']

    # Load sentences
    train_sentences = load_sentences(train_path, lower, zeros)

    # Use selected tagging scheme
    check_tag_chunking(train_sentences)

    # Create a dictionary / mapping of words
    # If we use pretrained embeddings, we add them to the dictionary.
    if parameters['pre_emb']:
        dico_words_train = word_mapping(train_sentences, lower, vocabulary_size)[0]
        dico_words, word_to_id, id_to_word = augment_with_pretrained(
            dico_words_train.copy(),
            parameters['pre_emb'],
            list(itertools.chain.from_iterable(
                [[w[0] for w in s] for s in dev_sentences + test_sentences])
            ) if not parameters['all_emb'] else None
        )
    else:
        #{word: number}
        dico_words, word_to_id, id_to_word = word_mapping(train_sentences, 
                                                lower,vocabulary_size)
        dico_words_train = dico_words

    # Create a dictionary and a mapping for tags
    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

    # index data
    train_data = prepare_dataset(
        train_sentences, word_to_id, tag_to_id, lower
    )

    print("%i sentences in train ."%(len(train_data)))

    # Save the mappings to disk
    print('Saving the mappings to disk...')
    save_mappings(id_to_word, id_to_tag, parameters['save_emb'])

    return train_data, len(tag_to_id)