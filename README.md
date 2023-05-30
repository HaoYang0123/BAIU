## BAIU

We use the [DeepCTR-Torch public model](https://github.com/shenweichen/DeepCTR-Torch), and add BERT/NLP sub-models into this module. 


## Introduction

This is the source code of BERT Attention method based on both Item and User information (named BAIU), an approach of extracting NLP features for CTR task from [Shopee](https://shopee.co.id/). 


## Requirements and Installation
We recommended the following dependencies.

* Python 3.8
* [PyTorch](http://pytorch.org/) 1.8.0
* Details shown in requirements.txt

The network structure of BAIU

<img src="https://github.com/HaoYang0123/BAIU/blob/main/workflow.png" width="745" alt="workflow" />


## Download data
1. KDD public data set can be download from this [link](https://www.kaggle.com/c/kddcup2012-track2).
2. NLP features for KDD public data set can be download from these following links:
* BAIU NLP features is [here](https://drive.google.com/file/d/1A29WTDiHndC9yRnuCsLtyWSSSAC_gWqG/view?usp=share_link).
* word2vec embeddings is [here](https://drive.google.com/file/d/1y2IsGhrGK6JN7Obc6BDPJ8eRvtokggN7/view?usp=share_link).
* tf-idf keywords is [here](https://drive.google.com/file/d/1EFnvjOgpji40Q3_78MbLAKKXvudT6BoF/view?usp=share_link).


## Training new models for public KDD data
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

nlp_path="${kdd_data_root}/titleid_tokensid.txt" # Note that: there are three NLP features: title/description/query, then you need run this script three times.
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
#!/bin/bash

set -x
cd examples
train_path=$1  # kdd_2012/track2/training.csv
outpath=$2     # ./pred.txt

# Run Wide & Deep model
python -u run_classification_kdd_wd_bert.py ${train_path} -1 ${outpath}

# Run xDeepFM model
python -u run_classification_kdd_xdfm_bert.py ${train_path} -1 ${outpath}
```

Step4. To evaluate the AUC and RIG 
```bash
#!/bin/bash

set -x



cd evaluate
inpath=$1  # ./pred.txt

# to get the AUC and RIG on all data
python -u get_auc.py ${inpath}

# to get the AUC and RIG on tail data
python -u get_auc.py ${inpath} 10
```

## Train models for Shopee data
Step1. To train BERT models with LM loss:
```bash
#!/bin/bash
set -x

cd ./nlp_sim_public

country=BR
sample_ratio=1.75
title_path=./sample_data/BR_title_info.txt
lm_model_folder=./sample_data/models_lm_${country}

bert_name=neuralmind/bert-base-portuguese-cased
if [[ "$country" == "ID" ]]
then
    bert_name=cahya/distilbert-base-indonesian
elif [[ "$country" == "MY" ]]
then
    bert_name=cahya/distilbert-base-indonesian
elif [[ "$country" == "SG" ]]
then
    bert_name=cahya/distilbert-base-indonesian
elif [[ "$country" == "TW" ]]
then
    bert_name=uer/chinese_roberta_L-6_H-768
elif [[ "$country" == "BR" ]]
then
    bert_name=neuralmind/bert-base-portuguese-cased
elif [[ "$country" == "PH" ]]
then
    bert_name=jcblaise/bert-tagalog-base-cased
elif [[ "$country" == "VN" ]]
then
    bert_name=trituenhantaoio/bert-base-vietnamese-uncased
elif [[ "$country" == "TH" ]]
then
    bert_name=monsoon-nlp/bert-base-thai
fi
echo ${bert_name}

python -u train_lm.py \
    --title-path ${title_path} \
    --batch-size 128 \
    --sample-ratio ${sample_ratio} \
    --workers 4 \
    --epoches 1 \
    --learning-rate 5e-5 \
    --model-folder ${lm_model_folder} \
    --print-freq 10 \
    --bert-model ${bert_name} \
    --max-seq-len 50
```

Step2. To predict BAIU NLP features:
```bash
#!/bin/bash
set -x

cd nlp_sim_public

country=BR
title_path=./sample_data/BR_title_info.txt
model_path=./sample_data/models_lm_${country}/nlp_lm_checkpoint_0.pt
out_nlp_feature_folder=./sample_data/item_feat_${country}

bert_name=neuralmind/bert-base-portuguese-cased
if [[ "$country" == "ID" ]]
then
    bert_name=cahya/distilbert-base-indonesian
elif [[ "$country" == "MY" ]]
then
    bert_name=cahya/distilbert-base-indonesian
elif [[ "$country" == "SG" ]]
then
    bert_name=cahya/distilbert-base-indonesian
elif [[ "$country" == "TW" ]]
then
    bert_name=uer/chinese_roberta_L-6_H-768
elif [[ "$country" == "BR" ]]
then
    bert_name=neuralmind/bert-base-portuguese-cased
elif [[ "$country" == "PH" ]]
then
    bert_name=jcblaise/bert-tagalog-base-cased
elif [[ "$country" == "VN" ]]
then
    bert_name=trituenhantaoio/bert-base-vietnamese-uncased
elif [[ "$country" == "TH" ]]
then
    bert_name=monsoon-nlp/bert-base-thai
fi
echo ${bert_name}

python -u predict_for_item_speed.py \
    --title-path ${title_path} \
    --outfolder ${out_nlp_feature_folder} \
    --checkpoint-path ${model_path} \
    --batch-size 512 \
    --model-flag ori \
    --workers 6 \
    --print-freq 10 \
    --topk 5 \
    --his-browse-num 30 \
    --his-add-num 10 \
    --his-buy-num 10 \
    --bert-model ${bert_name} \
    --max-seq-len 120
```

Step3. To train CTR model with NLP features:
```bash
#!/bin/bash
set -x

cd ./nlp_sim_public

country=BR
model_path=./sample_data/models_lm_${country}/nlp_lm_checkpoint_0.pt
title_path=./sample_data/BR_title_info.txt
ctr_path=./sample_data/BR_ctr_info.txt
nlp_feature_folder=./sample_data/item_feat_${country}
pred_out_path=./sample_data/pred.txt
test_title_path=./sample_data/BR_title_info.txt  # test data
test_ctr_path=./sample_data/BR_ctr_info.txt      # test data
out_ctr_model_folder=./sample_data/models_ctr_${country}

bert_name=neuralmind/bert-base-portuguese-cased
if [[ "$country" == "ID" ]]
then
    bert_name=cahya/distilbert-base-indonesian
elif [[ "$country" == "MY" ]]
then
    bert_name=cahya/distilbert-base-indonesian
elif [[ "$country" == "SG" ]]
then
    bert_name=cahya/distilbert-base-indonesian
elif [[ "$country" == "TW" ]]
then
    bert_name=uer/chinese_roberta_L-6_H-768
elif [[ "$country" == "BR" ]]
then
    bert_name=neuralmind/bert-base-portuguese-cased
elif [[ "$country" == "PH" ]]
then
    bert_name=jcblaise/bert-tagalog-base-cased
elif [[ "$country" == "VN" ]]
then
    bert_name=trituenhantaoio/bert-base-vietnamese-uncased
elif [[ "$country" == "TH" ]]
then
    bert_name=monsoon-nlp/bert-base-thai
fi
echo ${bert_name}

python -u train_noquery_nobasic_speed_sim2.py \
    --title-path ${title_path} \
    --npy-folder ${nlp_feature_folder} \
    --fc-hidden-size 32 \
    --pred-outpath ${pred_out_path} \
    --test-title-path ${test_title_path} \
    --ctr-path ${ctr_path} \
    --test-ctr-path ${test_ctr_path} \
    --checkpoint-path ${model_path} \
    --checkpoint-flag v4 \
    --batch-size 1024 \
    --workers 4 \
    --epoches 1 \
    --learning-rate 0.0 \
    --lr-attention 0.0005 \
    --model-folder ${out_ctr_model_folder} \
    --print-freq 10 \
    --split 10 \
    --eval-freq 0.1 \
    --his-browse-num 30 \
    --his-add-num 10 \
    --his-buy-num 10 \
    --his-sim-num 150 \
    --fix-bert \
    --bert-model ${bert_name} \
    --max-seq-len 120

```
