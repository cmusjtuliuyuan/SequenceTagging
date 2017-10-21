#!/bin/bash
####################################
#
# Dive the whole training to 50 part,
# Because there is some bug in my GPU memory
#
####################################
python main.py
python main.py --load models/model1.mdl
python main.py --load models/model2.mdl
python main.py --load models/model3.mdl
python main.py --load models/model4.mdl
python main.py --load models/model5.mdl
python main.py --load models/model6.mdl