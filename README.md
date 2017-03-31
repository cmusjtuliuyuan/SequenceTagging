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

![image](https://github.com/cmusjtuliuyuan/SequenceTagging/blob/master/CRFLSTM.png)

I am doing the pre-training part now....
After pre-training, I will calculate the derivation by hand to speed up...

Some of the code is from https://github.com/glample/tagger
I want to make the structure of my code is similar with /glample/tagger as much as possible.

We will use the following dataset now, text chunking

