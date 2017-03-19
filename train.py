import optparse
import os
from collections import OrderedDict
from loader import load_train_step_datasets, load_test_step_datasets
import model
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import evaluate

#TO DO add Dictionary size

optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="data/eng.train",
    help="Train set location"
)
optparser.add_option(
    "-D", "--dev", default="data/eng.testa",
    help="Development dataset"
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
Parse_parameters['train'] = opts.train
Parse_parameters['vocab_size'] = opts.vocab_size

# Check parameters validity
assert os.path.isfile(opts.train)

# load datasets
train_data, tagset_size, dictionaries = load_train_step_datasets(Parse_parameters)
dev_data = load_test_step_datasets(Parse_parameters, opts.dev, dictionaries)


#embedding_dim, hidden_dim, vocab_size, tagset_size
# Model parameters
Model_parameters = OrderedDict()
Model_parameters['vocab_size'] = opts.vocab_size
Model_parameters['embedding_dim'] = opts.embedding_dim
Model_parameters['hidden_dim'] = opts.hidden_dim
Model_parameters['tagset_size'] = tagset_size


model = model.LSTMTagger(Model_parameters)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

n_epochs = 1  # number of epochs over the training set
freq_eval = 1  # evaluate on dev every freq_eval steps
count = 0

for epoch in xrange(n_epochs): # again, normally you would NOT do 300 epochs, it is toy data
		epoch_costs = []
		print("Starting epoch %i..." % (epoch))
		for i, index in enumerate(np.random.permutation(len(train_data))):
				# Step 1. Remember that Pytorch accumulates gradients.  We need to clear them out
				# before each instance
				model.zero_grad()
    
				# Step 2. Get our inputs ready for the network, that is, turn them into Variables
				# of word indices.
				sentence_in = autograd.Variable(torch.LongTensor(train_data[index]['words']))
				targets = autograd.Variable(torch.LongTensor(train_data[index]['tags']))

				# Step 3. Run our forward pass.
				tag_scores = model(sentence_in)

				# Step 4. Compute the loss, gradients, and update the parameters by calling
				# optimizer.step()
				loss = loss_function(tag_scores, targets)
				epoch_costs.append(loss.data.numpy())
				loss.backward()
				optimizer.step()

				#Evaluate on development set and test set
				if count % freq_eval == 0:
							dev_score = evaluate(parameters, f_eval, dev_sentences,
                        dev_data, id_to_tag, dico_tags)
				
		print("Epoch %i, cost average: %f" % (epoch, np.mean(epoch_costs)))

print("haha")