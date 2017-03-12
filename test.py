import optparse
import os
import loader
from collections import OrderedDict
from loader import check_tag_chunking
from loader import word_mapping, tag_mapping
from loader import prepare_dataset
from loader import augment_with_pretrained
from loader import save_mappings



optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="data/yuantest",
    help="Train set location"
)
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-p", "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-s", "--save_emb", default="embedding/temp",
    help="Location of pretrained embeddings"
)
opts = optparser.parse_args()[0]

# Parse parameters
parameters = OrderedDict()
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['pre_emb'] = opts.pre_emb
parameters['save_emb'] = opts.save_emb

# Check parameters validity
assert os.path.isfile(opts.train)

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']

# Load sentences
train_sentences = loader.load_sentences(opts.train, lower, zeros)

# Use selected tagging scheme
check_tag_chunking(train_sentences)

# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
if parameters['pre_emb']:
    dico_words_train = word_mapping(train_sentences, lower)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    #{word: number}
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
    dico_words_train = dico_words

# Create a dictionary and a mapping for tags
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

# index data
train_data = prepare_dataset(
    train_sentences, word_to_id, tag_to_id, lower
)

print "%i sentences in train ." % (
    len(train_data))

# Save the mappings to disk
print 'Saving the mappings to disk...'
save_mappings(id_to_word, id_to_tag, parameters['save_emb'])

print(train_data)