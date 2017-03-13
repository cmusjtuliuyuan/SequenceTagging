import optparse
import os
from collections import OrderedDict
from loader import load_train_step_datasets




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
parameters['train']=opts.train

# Check parameters validity
assert os.path.isfile(opts.train)

# load datasets
train_data = load_train_step_datasets(parameters)
print(train_data)