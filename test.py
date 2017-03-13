import optparse
import os
from collections import OrderedDict
from loader import load_train_step_datasets
import model


#TO DO add Dictionary size

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
optparser.add_option(
    "-v", "--vocab_size", default="2000",
    type='int', help="vocab_size"
)
optparser.add_option(
    "-e", "--embedding_dim", default="100",
    type='int', help="words hidden dimension"
)
optparser.add_option(
    "-d", "--hidden_dim", default="100",
    type='int', help="LSTM hidden dimension"
)
opts = optparser.parse_args()[0]

# Parse parameters
Parse_parameters = OrderedDict()
Parse_parameters['lower'] = opts.lower == 1
Parse_parameters['zeros'] = opts.zeros == 1
Parse_parameters['pre_emb'] = opts.pre_emb
Parse_parameters['save_emb'] = opts.save_emb
Parse_parameters['train']=opts.train
Parse_parameters['vocab_size']=opts.vocab_size

# Check parameters validity
assert os.path.isfile(opts.train)

# load datasets
train_data, tagset_size= load_train_step_datasets(Parse_parameters)
print(train_data[0]['words'])

#embedding_dim, hidden_dim, vocab_size, tagset_size
# Model parameters
Model_parameters = OrderedDict()
Model_parameters['vocab_size'] = opts.vocab_size
Model_parameters['embedding_dim'] = opts.embedding_dim
Model_parameters['hidden_dim'] = opts.hidden_dim
Model_parameters['tagset_size'] = tagset_size

model = model.LSTMTagger(Model_parameters)


