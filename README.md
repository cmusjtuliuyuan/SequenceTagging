# SequenceTagging
DAP project, PyTorch, LSTM CRF model

Support Mini-batch now

This is for Yuan's Data Analysis Project. 
I want to implment LSTM-CRF autoencoder in PyTorch.

Please use Pytorch 0.2.0!!!
Thanks: https://github.com/kaniblu/pytorch-bilstmcrf https://github.com/spro/practical-pytorch https://github.com/rguthrie3/DeepLearningForNLPInPytorch https://github.com/glample/tagger My code is based on them. 

The data pre-processing part is done.

LSTM in PyTorch training part is done.

Evaluation part is done.

The CRF-LSTM model part is done.

Test this model on CONLL2000, the result shown in below:

Pre-Training part is done.
The pre-training dataset can be downloaded from https://nlp.stanford.edu/projects/glove/ 

The accuracy after pre-training is 92.88%, before pre-training is 90.96%

The marginal decode is done.

The maximum labelwise accuracy part is done.

Split LSTMCRF into two models. Add CRF module.

Add some hand engineer.

The none-pretrain SGD F1 score ~ 88.5

The pretrain SGD F1 score ~ 91.2
