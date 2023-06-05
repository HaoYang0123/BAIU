# -*- coding: utf-8 -*-
import argparse
import json
import os
import copy
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from transformers import BertTokenizer, BertModel

import sys
sys.path.insert(0, '..')
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
#from deepctr_torch.models.deepfm import DeepFM
#from deepctr_torch.models.wdl import WDL
from deepctr_torch.models.xdeepfm_bert_baiu import xDeepFM

def _load_title(path, tokenid2new_id, tokenizer):
    id2id_list = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            tmp = line.strip('\n').split('\t')
            id = int(tmp[0])
            id_in_title = [tokenid2new_id.get(v, 0) for v in tmp[1].split('|')]
            new_id_list = tokenizer.convert_tokens_to_ids(['[CLS]']) + id_in_title + \
                          tokenizer.convert_tokens_to_ids(['[SEP]'])
            id2id_list[id] = new_id_list
    return id2id_list

def get_id_and_mask(data, name, titleid2id_list, max_len):
    new_title_id_list = []
    new_title_mask_list = []
    for one_value in data[name]:
        one_id_list = titleid2id_list.get(int(one_value), [])[:max_len]
        one_mask_list = [1 for _ in range(len(one_id_list))]
        one_mask_list += [0 for _ in range(max_len - len(one_id_list))]
        one_id_list += [0 for _ in range(max_len - len(one_id_list))]
        new_title_id_list.append(one_id_list)
        new_title_mask_list.append(one_mask_list)
    data[name+"_id"] = new_title_id_list
    data[name+"_mask"] = new_title_mask_list
    return [name+"_id", name+"_mask"]

def load_user_session(user_path):
    userid2ad_list = {}
    with open(user_path, encoding='utf8') as f:
        for line in f:
            d=json.loads(line.strip('\n'))
            userid = d['userid']
            ad_list = d['ads']
            userid2ad_list[userid] = ad_list
    return userid2ad_list

if __name__ == "__main__":
    param = argparse.ArgumentParser(description='Run kdd data')
    param.add_argument("--input_path", type=str,
                       default=r"track2/training.csv", help="input path of training.csv")
    param.add_argument("--gpu_id", type=int,
                       default=-1, help="gpu id (not used)")
    param.add_argument("--outpath", type=str,
                       default=r"pred.txt", help="outpath")
    param.add_argument("--nlp_feature_folder", type=str,
                       default=r"", help="nlp folder")
    param.add_argument("--user_ad_path", type=str,
                       default=r"track2/userid_adid_merge.txt", help="user ad path")
    config = param.parse_args()
    input_path = config.input_path
    gpu_id = config.gpu_id
    outpath = config.outpath
    nlp_folder = config.nlp_feature_folder
    if not os.path.exists(nlp_folder):
        print("input nlp feature folder not exists", nlp_folder)
        sys.exit(-1)
    title_npy_folder = os.path.join(nlp_folder, "title")
    query_npy_folder = os.path.join(nlp_folder, "query")
    des_npy_folder = os.path.join(nlp_folder, "des")
    if not os.path.exists(title_npy_folder) or not os.path.exists(query_npy_folder) or not os.path.exists(des_npy_folder):
        print("input nlp title/query/des folder not exists")
        sys.exit(-1)
    user_ad_path = config.user_ad_path
    if not os.path.exists(user_ad_path):
        print("input user ad path not exists", user_ad_path)
        sys.exit(-1)

    print("GPU_ID", gpu_id)
    print("outpath", outpath)
    sparse_features = ['C' + str(i) for i in range(1, 13)]

    bert_model_name = "bert-base-uncased"

    userid2ad_list = load_user_session(user_ad_path)
    print("#users", len(userid2ad_list))

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    userid_column_name = "C10"
    adid_column_name = "C2"
    title_column_name = "C8"
    query_column_name = "C6"
    des_column_name = "C9"
    need_feat_set = set([title_column_name, query_column_name, des_column_name, userid_column_name, adid_column_name])
    title_max_len = 20
    query_max_len = 10
    des_max_len = 30

    int2str_dict = {}
    for name in sparse_features:
        int2str_dict[name] = str

    data = pd.read_csv(input_path)  #, converters=int2str_dict)


    data[sparse_features] = data[sparse_features].fillna(0, )  #'-1', )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    token_features, token_size_list = [], []
    for feat in sparse_features:
        print("start feat", feat)
        lbe = LabelEncoder()
        #print("ori", feat, data[feat])
        if feat in need_feat_set:
            print("start copy ", feat)
            data[feat+"_ori"] = copy.deepcopy(data[feat])
            token_features.append(feat+"_ori")
        data[feat] = lbe.fit_transform(data[feat])
        #print("fff", feat, data[feat])
    #mms = MinMaxScaler(feature_range=(0, 1))
    #data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 10, embedding_dim=4)
                              for feat in sparse_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    print("dnn feature", dnn_feature_columns)
    print("linear features", linear_feature_columns)

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)
    print("feature names", feature_names)
    print("token names", token_features)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.1, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names+token_features}
    test_model_input = {name: test[name] for name in feature_names+token_features}
    #print("train_model_input", train_model_input)
    #print("test_model_input", test_model_input)

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    #model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #               task='binary',
    #               l2_reg_embedding=1e-5, device=device)
    #model = WDL(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, 
    #                task='binary', 
    #                l2_reg_embedding=1e-5, device=device)
    model = xDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                        task='binary',
                        l2_reg_embedding=1e-5, device=device, tokenizer=tokenizer,
                   title_npy_folder=title_npy_folder, query_npy_folder=query_npy_folder, des_npy_folder=des_npy_folder, userid2ad_list=userid2ad_list)

    print("Model**********", model)
    model.compile("adagrad", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )

    print("start fit======")
    history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=1, verbose=2,
                        validation_split=0.0)
    print("start predict")
    pred_ans, pred_info_list = model.predict(test_model_input, None, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

    with open(outpath, 'w', encoding='utf8') as fw:
        for (score, label, info_list) in zip(pred_ans, test[target].values, pred_info_list):
            fw.write(str(score[0])+'\t'+str(label[0])+'\t'+'\t'.join(map(str, info_list))+'\n')
