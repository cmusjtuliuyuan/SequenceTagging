import optparse
import os
from collections import OrderedDict
from loader import prepare_dictionaries, get_word_embedding_matrix
import LstmCrfModel
import LstmModel
import torch
import numpy as np
from utils import save_model_dictionaries, load_parameters
from train import train
import cPickle

optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="data/conll2000.train.txt",
    help="Train set location"
)
optparser.add_option(
    "-D", "--dev", default="data/conll2000.test.txt",
    help="Development dataset"
)
optparser.add_option(
    "-l", "--lower", default="1",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="1",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-p", "--pre_emb", default=None,#'embedding/glove.6B.100d.txt',
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-v", "--vocab_size", default="8000",
    type='int', help="vocab_size"
)
optparser.add_option(
    "-e", "--embedding_dim", default="100",
    type='int', help="words hidden dimension"
)
optparser.add_option(
    "-d", "--hidden_dim", default="200",
    type='int', help="LSTM hidden dimension"
)
optparser.add_option(
    "-c", "--clip", default=5.0,
    help="gradient clipping l2 norm"
)
optparser.add_option(
    "-f", "--freeze", default=False,
    help="Wheter freeze the embedding layer or not"
)
optparser.add_option(
    "-s", "--save", default='model',
    help="Model and dictionareis stored postition"
)
optparser.add_option(
    "--load", default=None,
    help="Load pre-trained Model and dictionaries"
)

# TODO delete lower

def main():
    np.random.seed(15213)
    torch.manual_seed(15213)
    opts = optparser.parse_args()[0]

    # Parse parameters
    Parse_parameters = OrderedDict()
    Parse_parameters['lower'] = opts.lower == 1
    Parse_parameters['zeros'] = opts.zeros == 1
    Parse_parameters['pre_emb'] = opts.pre_emb
    Parse_parameters['train'] = opts.train
    Parse_parameters['development'] = opts.dev
    Parse_parameters['vocab_size'] = opts.vocab_size

    # Check parameters validity
    assert os.path.isfile(opts.train)
    assert os.path.isfile(opts.dev)
    if opts.pre_emb:
        assert opts.embedding_dim in [50, 100, 200, 300]
        assert opts.lower == 1

    # load datasets
    if not opts.load:
        dictionaries = prepare_dictionaries(Parse_parameters)
    else:
        # load dictionaries
        with open(opts.load+'/dictionaries.dic', 'rb') as f:
            dictionaries = cPickle.load(f)
        # load parameters
        opts = load_parameters(opts.load, opts)

    # Model parameters
    Model_parameters = OrderedDict()
    Model_parameters['vocab_size'] = opts.vocab_size
    Model_parameters['embedding_dim'] = opts.embedding_dim
    Model_parameters['hidden_dim'] = opts.hidden_dim
    Model_parameters['tagset_size'] = len(dictionaries['tag_to_id'])
    Model_parameters['freeze'] = opts.freeze


    #model = LstmModel.LSTMTagger(Model_parameters)
    model = LstmCrfModel.LSTM_CRF(Model_parameters)
    # gradients are allocated lazily, so they are not shared here
    model.share_memory()

    # If using pre-train, we need to initialize word-embedding layer
    if opts.pre_emb and not opts.load:
        print("Initialize the word-embedding layer")
        initial_matrix = get_word_embedding_matrix(dictionaries['word_to_id'], 
                    opts.pre_emb, opts.embedding_dim)
        model.init_word_embedding(initial_matrix)

    # Load pre-trained model
    if opts.load:
      model.load_state_dict(torch.load(opts.load+'/model.mdl'))
    
    train(model, Parse_parameters, opts, dictionaries)
    # Save model and dictionaries
    save_model_dictionaries('model', model, dictionaries, opts)



if __name__ == '__main__':
    main()

