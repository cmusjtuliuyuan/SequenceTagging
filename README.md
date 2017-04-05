# SequenceTagging
DAP project, PyTorch, LSTM model

This is for Yuan's Data Analysis Project. 
I want to implment LSTM-CRF autoencoder in PyTorch.

The data pre-processing part is done.

LSTM in PyTorch training part is done.

Evaluation part is done.

Test this model on CONLL2000, the result shown in below:
![image](https://github.com/cmusjtuliuyuan/SequenceTagging/blob/master/LSTM.png)

The CRF-LSTM model part is done.

Test this model on CONLL2000, the result shown in below:

![image](https://github.com/cmusjtuliuyuan/SequenceTagging/blob/master/no_pre_train_CRFLSTM.9096.png)

Pre-Training part is done.
The pre-training dataset can be downloaded from https://nlp.stanford.edu/projects/glove/ 

The accuracy after pre-training is 92.88%, before pre-training is 90.96%
![image](https://github.com/cmusjtuliuyuan/SequenceTagging/blob/master/pre_train_CRFLSTM.9288.png)

#TODO Discuss the lowercase issue. and tag_size+2 issue.
After pre-training, I will calculate the derivation by hand to speed up...
