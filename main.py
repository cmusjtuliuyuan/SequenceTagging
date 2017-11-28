import optparse
import os
from collections import OrderedDict
from loader import prepare_dictionaries, get_word_embedding_matrix, get_senna_embedding_matrix
import Autoencoder

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
    "-p", "--pre_emb", default= None, #'senna',
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-v", "--vocab_size", default="8000",
    type='int', help="vocab_size"
)
optparser.add_option(
    "-e", "--embedding_dim", default="50",
    type='int', help="words hidden dimension"
)
optparser.add_option(
    "-c", "--char_dim", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-d", "--hidden_dim", default="300",
    type='int', help="LSTM hidden dimension"
)
optparser.add_option(
    "--clip", default=5.0,
    help="gradient clipping l2 norm"
)
optparser.add_option(
    "-f", "--freeze", default=False,
    help="Wheter freeze the embedding layer or not"
)
optparser.add_option(
    "--load", default=None,#'models/model2.mdl',
    help="Load pre-trained Model and dictionaries"
)
optparser.add_option(
    "--type", default='supervised',
    type = 'string', help="Do supervised training or not"
)
optparser.add_option(
    "--store", default=None,
    type = 'string', help="Where to store the model"
)
optparser.add_option(
    "--unsupervised_path",
    type = 'string', help="Where to load the unsupervised data"
)
# TODO delete lower

def main():
    np.random.seed(15213)
    torch.manual_seed(15213)
    opts = optparser.parse_args()[0]
    CUDA_AVAILABLE = torch.cuda.is_available()

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
    assert opts.type == 'supervised' or 'unsupervised' in opts.type
    if opts.pre_emb:
        assert opts.embedding_dim in [50, 100, 200, 300]
        assert opts.lower == 1

    dictionaries = prepare_dictionaries(Parse_parameters)

    # Model parameters
    Model_parameters = OrderedDict()
    Model_parameters['vocab_size'] = opts.vocab_size
    Model_parameters['embedding_dim'] = opts.embedding_dim
    Model_parameters['hidden_dim'] = opts.hidden_dim
    Model_parameters['char_dim'] = opts.char_dim
    Model_parameters['char_lstm_dim'] = opts.char_lstm_dim
    Model_parameters['tagset_size'] = len(dictionaries['tag_to_id'])
    Model_parameters['char_size'] = len(dictionaries['char_to_id'])
    Model_parameters['freeze'] = opts.freeze
    Model_parameters['cuda'] = CUDA_AVAILABLE


    #model = LstmModel.LSTMTagger(Model_parameters)
    model = Autoencoder.Autoencoder(Model_parameters)
    if CUDA_AVAILABLE:
        model = model.cuda()
    # gradients are allocated lazily, so they are not shared here
    #model.share_memory()

    # If using pre-train, we need to initialize word-embedding layer
    if opts.pre_emb and not opts.load:
        print("Initialize the word-embedding layer")
        if 'glove' in opts.pre_emb:
            initial_matrix = get_word_embedding_matrix(dictionaries['word_to_id'], 
                    opts.pre_emb, opts.embedding_dim)
        else:
            assert 'senna' in opts.pre_emb
            initial_matrix = get_senna_embedding_matrix(dictionaries['word_to_id'])
        model.init_word_embedding(initial_matrix)

    if opts.load:
        model.load_state_dict(torch.load(opts.load))
        print 'load:', opts.load

    train(model, Parse_parameters, opts, dictionaries)

    if opts.store:
        torch.save(model.state_dict(), opts.store)
        print 'save model in: %s'%(opts.store,)


if __name__ == '__main__':
    main()

