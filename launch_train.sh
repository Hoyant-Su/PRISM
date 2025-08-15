#!/bin/bash
export PYTHONPATH="./PRISM"

##stage I
python ssl_pretraining/train/train_distill.py

##stage II
python ssl_pretraining/train/train_codebook.py
python ssl_pretraining/train/train_projector.py

##stage III
bash finetune_survival/CTSL_Cox/train_Model_Cox.sh

