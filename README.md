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
Step1. To train BERT models with LM loss, note that you have downloaded the BAIU NLP features, then Step1 and Step2 are not needed: 
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
Step2. To predict BAIU NLP features:
```bash
#!/bin/bash

set -x
cd nlp_kdd
kdd_data_root=$1  # kdd_2012/track2
model_path=$2  # ./models/nlp_lm_checkpoint_0.pt
out_nlp_feature_root=$3  # ./embedding/[title/description/query]

nlp_path="${kdd_data_root}/titleid_tokensid.txt"
python -u predict_embedding.py \
    --title-path ${nlp_path} \
    --tokenid2id-path ./tokenid2bertid.json \
    --batch-size 128 \
    --epoches 1 \
    --bert-model bert-base-uncased \
    --checkpoint-path ${model_path} \
    --out-folder ${out_nlp_feature_root} \
    --mask-txt-ratio 0.1 \
    --max-mask-num 3 \
    --max-seq-len 100 \
    --print-freq 10

```

Step3. To train CTR model with NLP features:
```bash
```
