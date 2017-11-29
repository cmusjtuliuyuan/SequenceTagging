import codecs
import re
import os

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


path = 'data/wiki100_unlabel'
files = os.listdir(path)
for file in files:
	if file != '.DS_Store':
		with codecs.open('data/wiki100/'+file+'.txt', 'w', 'utf8') as f:
			for sentence in codecs.open('data/wiki100_unlabel/'+file, 'r', 'utf8'):
			    sentence= zero_digits(sentence.rstrip()).split()
			    if len(sentence) >= 5 and len(sentence) <= 80:
			    	for word in sentence:
			    		f.write(word+'\n')
			    	f.write('\n')
		  	


