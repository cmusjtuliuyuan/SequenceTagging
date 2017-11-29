#!/bin/bash
####################################
#
# Dive the whole training to 50 part,
# Because there is some bug in my GPU memory
#
####################################
#python main.py --type supervised --store model/susmodel.mdl
#ck
python main.py --type unsupervised --unsupervised_path data/wiki200/wiki200aa.txt --load model/susmodel.mdl --store model/susmodel.mdl
python main.py --type unsupervised --unsupervised_path data/wiki200/wiki200ab.txt --load model/susmodel.mdl --store model/susmodel.mdl
python main.py --type unsupervised --unsupervised_path data/wiki200/wiki200ac.txt --load model/susmodel.mdl --store model/susmodel.mdl


python main.py --type supervised --load model/susmodel.mdl --store model/model2.mdl

python main.py --type unsupervised --unsupervised_path data/wiki200/wiki200ad.txt --load model/susmodel.mdl --store model/susmodel.mdl
python main.py --type unsupervised --unsupervised_path data/wiki200/wiki200ae.txt --load model/susmodel.mdl --store model/susmodel.mdl
python main.py --type unsupervised --unsupervised_path data/wiki200/wiki200af.txt --load model/susmodel.mdl --store model/susmodel.mdl
python main.py --type unsupervised --unsupervised_path data/wiki200/wiki200ag.txt --load model/susmodel.mdl --store model/susmodel.mdl
python main.py --type unsupervised --unsupervised_path data/wiki200/wiki200ah.txt --load model/susmodel.mdl --store model/susmodel.mdl
python main.py --type unsupervised --unsupervised_path data/wiki200/wiki200ai.txt --load model/susmodel.mdl --store model/susmodel.mdl
python main.py --type unsupervised --unsupervised_path data/wiki200/wiki200aj.txt --load model/susmodel.mdl --store model/susmodel.mdl
python main.py --type unsupervised --unsupervised_path data/wiki200/wiki200ak.txt --load model/susmodel.mdl --store model/susmodel.mdl

python main.py --type supervised --load model/susmodel.mdl --store model/model6.mdl