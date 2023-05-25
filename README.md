## BAIU

We use the [DeepCTR-Torch public model](https://github.com/shenweichen/DeepCTR-Torch), and add BERT/NLP sub-models into this module. 


## Introduction

This is the source code of BERT Attention method based on both Item and User information (named BAIU), an approach of extracting NLP features for CTR task from [Shopee](https://shopee.co.id/). 


## Requirements and Installation
We recommended the following dependencies.

* Python 3.8
* [PyTorch](http://pytorch.org/) 1.8.0
* [NumPy](http://www.numpy.org/) 1.23.4
* Details shown in requirements.txt

The network structure of BAIU

<img src="https://github.com/HaoYang0123/BAIU/blob/main/workflow.png" width="745" alt="workflow" />


## Download data
1. KDD public data set can be download from this [link](https://www.kaggle.com/c/kddcup2012-track2).
2. NLP features for KDD public data set can be download from these following links:
* BAIU NLP features is [here]().
* word2vec embeddings is [here]().
* tf-idf keywords is [here]().


## Training new models
To train BERT models:
```bash
#!/bin/bash

set -x
cd nlp_kdd
kdd_data_root=$1  # kdd_2012/track2
out_model_root=$2  # ./models

nlp_path="${kdd_data_root}/titleid_tokensid.txt;${kdd_data_root}/descriptionid_tokensid.txt;${kdd_data_root}/queryid_tokensid.txt"
python -u train.py \
    --title-path $nlp_path \
    --tokenid2id-path ./tokenid2bertid.json \
    --batch-size 64 \
    --epoches 1 \
    --bert-model bert-base-uncased \
    --model-folder ${out_model_root} \
    --learning-rate 2e-5 \
    --mask-txt-ratio 0.1 \
    --max-mask-num 3 \
    --max-seq-len 100 \
    --print-freq 10
```
