import optparse
import os
from collections import OrderedDict
from loader import prepare_dictionaries, load_dataset, get_word_embedding_matrix
import LstmModel
import LstmCrfModel
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import evaluate, plot_result
from loader import CAP_DIM



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
    "-d", "--hidden_dim", default="100",
    type='int', help="LSTM hidden dimension"
)
optparser.add_option(
    "-t", "--decode_method", default="marginal",
    type='int', help="Choose viterbi or marginal to decode the output tag"
)
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
dictionaries = prepare_dictionaries(Parse_parameters)
tagset_size = len(dictionaries['tag_to_id'])

train_data = load_dataset(Parse_parameters, opts.train, dictionaries)
dev_data = load_dataset(Parse_parameters, opts.dev, dictionaries)


# Model parameters
Model_parameters = OrderedDict()
Model_parameters['vocab_size'] = opts.vocab_size
Model_parameters['embedding_dim'] = opts.embedding_dim
Model_parameters['hidden_dim'] = opts.hidden_dim
Model_parameters['tagset_size'] = tagset_size
Model_parameters['lower'] = opts.lower == 1
Model_parameters['decode_method'] = opts.decode_method


#model = LstmModel.LSTMTagger(Model_parameters)
model = LstmCrfModel.BiLSTM_CRF(Model_parameters)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# If using pre-train, we need to initialize word-embedding layer
if opts.pre_emb:
	  print("Initialize the word-embedding layer")
	  initial_matrix = get_word_embedding_matrix(dictionaries['word_to_id'], 
	  					  opts.pre_emb, opts.embedding_dim)
	  model.init_word_embedding(initial_matrix)

n_epochs = 10 # number of epochs over the training set

accuracys = []
precisions = []
recalls = []
FB1s =[]


for epoch in xrange(n_epochs): # again, normally you would NOT do 300 epochs, it is toy data
    epoch_costs = []

    # evaluate
    eval_result = evaluate(model, dev_data, dictionaries, opts.lower)
    accuracys.append(eval_result['accuracy'])
    precisions.append(eval_result['precision'])
    recalls.append(eval_result['recall'])
    FB1s.append(eval_result['FB1'])

    print("Starting epoch %i..." % (epoch))
    for i, index in enumerate(np.random.permutation(len(train_data))):
        # Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
        # before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into Variables
        # of word indices.
        input_words = autograd.Variable(torch.LongTensor(train_data[index]['words']))
        targets = autograd.Variable(torch.LongTensor(train_data[index]['tags']))

        # Step 3. Run our forward pass. We combine this step with get_loss function
        #tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by calling
        # first check whether we use lower this parameter
        if opts.lower == 1:
            # We first convert it to one-hot, then input
            input_caps = torch.LongTensor(train_data[index]['caps'])
            loss = model.get_loss(targets, input_words = input_words, input_caps = input_caps)
        else:
            loss = model.get_loss(targets, input_words = input_words)

        epoch_costs.append(loss.data.numpy())
        loss.backward()
        optimizer.step()
			
    print("Epoch %i, cost average: %f" % (epoch, np.mean(epoch_costs)))

# Final Evaluation after training
eval_result = evaluate(model, dev_data, dictionaries, opts.lower)
accuracys.append(eval_result['accuracy'])
precisions.append(eval_result['precision'])
recalls.append(eval_result['recall'])
FB1s.append(eval_result['FB1'])


print("Plot final result")
plot_result(accuracys, precisions, recalls, FB1s)
