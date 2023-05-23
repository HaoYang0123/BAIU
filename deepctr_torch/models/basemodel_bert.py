# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,weichenswc@163.com
    zanshuxun, zanshuxun@aliyun.com

"""
from __future__ import print_function
import os
import time
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList

from ..inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup
from ..layers import PredictionLayer
from ..layers.utils import slice_arrays
from ..callbacks import History
from .utils import AverageMeter, meter_to_str


class Linear(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, device='cpu'):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.device = device
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False,
                                                      device=device)

        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, 1, sparse=True) for feat in
        #              self.sparse_feature_columns}
        #         )
        # .to("cuda:1")
        for tensor in self.embedding_dict.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), 1).to(
                device))
            torch.nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):

        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list

        linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)
            if sparse_feat_refine_weight is not None:
                # w_{x,i}=m_{x,i} * w_i (in IFM and DIFM)
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1, keepdim=False)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(
                dense_value_list, dim=-1).matmul(self.weight)
            linear_logit += dense_value_logit

        return linear_logit

class BaseDatasetOri(Dataset):
    def __init__(self, x, y, tokenizer, titleid2id_list, queryid2id_list, desid2id_list,
                 max_query_len=10, max_title_len=20, max_des_len=30):
        self.x = x
        self.y = y
        self.tokenizer = tokenizer
        self.cls_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]  # int: 101
        self.sep_id = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]  # int: 102
        self.titleid2id_list = titleid2id_list
        self.queryid2id_list = queryid2id_list
        self.desid2id_list = desid2id_list
        self.max_query_len = max_query_len
        self.max_title_len = max_title_len
        self.max_des_len = max_des_len

    def __len__(self):
        return len(self.x)

    def _pad(self, id_list, max_len):
        one_id_list = id_list[:max_len]
        one_mask_list = [1 for _ in range(len(one_id_list))]
        one_mask_list += [0 for _ in range(max_len - len(one_id_list))]
        one_id_list += [0 for _ in range(max_len - len(one_id_list))]

        return one_id_list, one_mask_list

    def __getitem__(self, idx):
        # # Click Impression    AdURL        AdId      AdvId  Depth Pos  QId       KeyId    TitleId  DescId  UId
        x, y = self.x[idx], self.y[idx]
        tit_id, qry_id, des_id = int(x[-3]), int(x[-2]), int(x[-5])
        tit_id_list = self.titleid2id_list.get(tit_id, [self.cls_id, self.sep_id])
        qry_id_list = self.queryid2id_list.get(qry_id, [self.cls_id, self.sep_id])
        des_id_list = self.desid2id_list.get(des_id, [self.cls_id, self.sep_id])
        tit_id_list, tit_id_mask = self._pad(tit_id_list, self.max_title_len)  # shape: #tokens
        qry_id_list, qry_id_mask = self._pad(qry_id_list, self.max_query_len)
        des_id_list, des_id_mask = self._pad(des_id_list, self.max_des_len)
        return x, torch.LongTensor(tit_id_list), torch.LongTensor(tit_id_mask), \
            torch.LongTensor(qry_id_list), torch.LongTensor(qry_id_mask), \
            torch.LongTensor(des_id_list), torch.LongTensor(des_id_mask), y

class BaseDataset(Dataset):
    def __init__(self, x, ori_x, y, title_npy_folder, query_npy_folder, des_npy_folder, userid2ad_list):
        self.x = x
        self.ori_x = ori_x
        self.y = y
        self.title_npy_folder = title_npy_folder
        self.query_npy_folder = query_npy_folder
        self.des_npy_folder = des_npy_folder
        self.userid2ad_list = userid2ad_list
        self.max_num_in_user_session = 30
        self.top_k = 5
        self.token2new_id = self._load_token_meta()

    def _load_token_meta(self):
        path = "/data/apple.yang/ctr/kdd_2012/track2/tokenid2bertid.json"
        with open(path) as f:
            d=json.load(f)
        return d

    def _load_item_npy(self, itemid, npy_folder):
        cur_folder = os.path.join(npy_folder, str(itemid)[:5])
        if not os.path.exists(cur_folder):
            #print("Error", itemid, npy_folder)
            return np.random.random(768), np.array([0]*self.top_k, dtype=np.int32), np.array([0]*self.top_k, dtype=np.int32)
        cur_path = os.path.join(cur_folder, f"{itemid}.pkl")
        if not os.path.exists(cur_path):
            #print("Error", itemid, npy_folder)
            return np.random.random(768), np.array([0]*self.top_k, dtype=np.int32), np.array([0]*self.top_k, dtype=np.int32)
        with open(cur_path, 'rb') as f:
            d = pickle.load(f)
        
        token_idx, token_msk = self._parse_token_feature(d) 
        return d['fea'], token_idx, token_msk

    def _parse_token_feature(self, d):
        word_idx_list, word_msk_list = [], []
        all_token_num = len(d['tokens'])
        all_token_list = d['tokens']
        all_score_list = d['score'].tolist()[1:1+all_token_num]  # delete [CLS] and [SEP] score
        all_tok_score_list = []
        for tok, score in zip(all_token_list, all_score_list):
            all_tok_score_list.append([tok, score])
        sorted_list = sorted(all_tok_score_list, key=lambda x:x[-1], reverse=True)[:self.top_k]
        word_idx_list, word_msk_list = [], []
        for tidx, score in sorted_list:
            if score <= 0.0: break
            word_idx_list.append(self.token2new_id.get(tidx, 0))
            word_msk_list.append(1)
            if len(word_idx_list) >= self.top_k: break
        for idx in range(self.top_k - len(word_idx_list)):
            word_idx_list.append(0)
            word_msk_list.append(0)
        return np.array(word_idx_list), np.array(word_msk_list)
        

    def _load_session_npy(self, session_list):
        tit_fea_list, des_fea_list, qry_fea_list = [], [], []
        tit_tok_list, tit_msk_list, des_tok_list, des_msk_list, qry_tok_list, qry_msk_list = [], [], [], [], [], []
        tit_des_mask_list, item_mask_list = [], []

        for (itemid, titleid, desid, queryid) in session_list:
            has_tit_flag, has_des_flag, has_qry_flag = False, False, False
            one_tit_fea, one_des_fea, one_qry_fea = None, None, None
            one_tit_tok, one_tit_msk, one_des_tok, one_des_msk, one_qry_tok, one_qry_msk = None, None, None, None, None, None

            cur_folder = os.path.join(self.title_npy_folder, str(titleid)[:5])
            if os.path.exists(cur_folder):
                cur_path = os.path.join(cur_folder, f"{titleid}.pkl")
                if os.path.exists(cur_path):
                    with open(cur_path, 'rb') as f:
                        d = pickle.load(f)
                    one_tit_fea = d['fea']
                    has_tit_flag = True
                    one_tit_tok, one_tit_msk = self._parse_token_feature(d)
            
            des_cur_folder = os.path.join(self.des_npy_folder, str(desid)[:5])
            if os.path.exists(des_cur_folder):
                des_cur_path = os.path.join(des_cur_folder, f"{desid}.pkl")
                if os.path.exists(des_cur_path):
                    with open(des_cur_path, 'rb') as f:
                        d = pickle.load(f)
                    one_des_fea = d['fea']
                    has_des_flag = True
                    one_des_tok, one_des_msk = self._parse_token_feature(d)

            qry_cur_folder = os.path.join(self.query_npy_folder, str(queryid)[:5])
            if os.path.exists(qry_cur_folder):
                qry_cur_path = os.path.join(qry_cur_folder, f"{queryid}.pkl")
                if os.path.exists(qry_cur_path):
                    with open(qry_cur_path, 'rb') as f:
                        d = pickle.load(f)
                    one_qry_fea = d['fea']
                    has_qry_flag = True
                    one_qry_tok, one_qry_msk = self._parse_token_feature(d)

            if has_tit_flag or has_des_flag or has_qry_flag:
                item_mask_list.append(1)
                if one_tit_fea is None: one_tit_fea = np.random.random(768)
                if one_des_fea is None: one_des_fea = np.random.random(768)
                if one_qry_fea is None: one_qry_fea = np.random.random(768)
                if one_tit_tok is None: one_tit_tok = np.array([0]*self.top_k, dtype=np.int32)
                if one_tit_msk is None: one_tit_msk = np.array([0]*self.top_k, dtype=np.int32)
                if one_des_tok is None: one_des_tok = np.array([0]*self.top_k, dtype=np.int32)
                if one_des_msk is None: one_des_msk = np.array([0]*self.top_k, dtype=np.int32)
                if one_qry_tok is None: one_qry_tok = np.array([0]*self.top_k, dtype=np.int32)
                if one_qry_msk is None: one_qry_msk = np.array([0]*self.top_k, dtype=np.int32)
                tit_fea_list.append(one_tit_fea)
                des_fea_list.append(one_des_fea)
                qry_fea_list.append(one_qry_fea)
                tit_tok_list.append(one_tit_tok)
                tit_msk_list.append(one_tit_msk)
                des_tok_list.append(one_des_tok)
                des_msk_list.append(one_des_msk)
                qry_tok_list.append(one_qry_tok)
                qry_msk_list.append(one_qry_msk)
                tit_des_mask_list.append([int(has_tit_flag), int(has_des_flag), int(has_qry_flag)])
                if len(tit_fea_list) == self.max_num_in_user_session: break
        for idx in range(self.max_num_in_user_session - len(tit_fea_list)):
            item_mask_list.append(0)
            tit_fea_list.append(np.random.random(768))
            des_fea_list.append(np.random.random(768))
            qry_fea_list.append(np.random.random(768))
            tit_tok_list.append(np.array([0]*self.top_k, dtype=np.int32))
            tit_msk_list.append(np.array([0]*self.top_k, dtype=np.int32))
            des_tok_list.append(np.array([0]*self.top_k, dtype=np.int32))
            des_msk_list.append(np.array([0]*self.top_k, dtype=np.int32))
            qry_tok_list.append(np.array([0]*self.top_k, dtype=np.int32))
            qry_msk_list.append(np.array([0]*self.top_k, dtype=np.int32))
            tit_des_mask_list.append([0,0,0])
        return tit_fea_list, des_fea_list, qry_fea_list, tit_des_mask_list, item_mask_list, tit_tok_list, tit_msk_list, des_tok_list, des_msk_list, qry_tok_list, qry_msk_list

    def __getitem__(self, idx):
        x, ori_x, y = self.x[idx], self.ori_x[idx], self.y[idx]
        #tit_id, qry_id, des_id = int(x[-5]), int(x[-7]), int(x[-4])
        # C6: query, C8: title, C9: des
        ori_tit_id, ori_qry_id, ori_des_id = int(ori_x[1]), int(ori_x[0]), int(ori_x[2])
        user_id = int(ori_x[3])
        ad_id = int(ori_x[4])
        session_list = self.userid2ad_list.get(user_id, [])[:self.max_num_in_user_session+1]
        session_list = [v for v in session_list if v[0] != ad_id]  # [TODO] delete current ad
        #print("+++++++, title", tit_id, ori_tit_id)
        #print("+++++++, qry", qry_id, ori_qry_id)
        #print("+++++++, des", des_id, ori_des_id)
        start_time = time.time()
        tit_fea, tit_tok_fea, tit_msk_fea = self._load_item_npy(ori_tit_id, self.title_npy_folder)  # 768
        qry_fea, qry_tok_fea, qry_msk_fea = self._load_item_npy(ori_qry_id, self.query_npy_folder)  # 768
        des_fea, des_tok_fea, des_msk_fea = self._load_item_npy(ori_des_id, self.des_npy_folder)    # 768
        #print("load item npy", time.time()-start_time)
        s_tit_fea, s_des_fea, s_qry_fea, s_tit_des_msk, s_itm_msk, s_tit_tok, s_tit_msk, s_des_tok, s_des_msk, s_qry_tok, s_qry_msk = self._load_session_npy(session_list)
        #print("load session npy", time.time()-start_time)
        s_tit_fea = torch.FloatTensor(s_tit_fea)
        s_des_fea = torch.FloatTensor(s_des_fea)
        s_qry_fea = torch.FloatTensor(s_qry_fea)
        s_tit_des_msk = torch.LongTensor(s_tit_des_msk)
        s_itm_msk = torch.LongTensor(s_itm_msk)
        s_tit_tok = torch.LongTensor(s_tit_tok)
        s_tit_msk = torch.LongTensor(s_tit_msk)
        s_des_tok = torch.LongTensor(s_des_tok)
        s_des_msk = torch.LongTensor(s_des_msk)
        s_qry_tok = torch.LongTensor(s_qry_tok)
        s_qry_msk = torch.LongTensor(s_qry_msk)
        #print("---FEA", s_tit_fea.shape, s_des_fea.shape, s_qry_fea.shape)
        #print("msk", s_tit_des_msk.shape)
        #print("itm msk", s_itm_msk.shape)
        #print("session_list", session_list)
        #print("----", s_tit_des_msk)
        return x, torch.FloatTensor(tit_fea), torch.FloatTensor(qry_fea), torch.FloatTensor(des_fea), \
            torch.LongTensor(tit_tok_fea), torch.LongTensor(tit_msk_fea), torch.LongTensor(qry_tok_fea), torch.LongTensor(qry_msk_fea), \
            torch.LongTensor(des_tok_fea), torch.LongTensor(des_msk_fea), \
            s_tit_fea, s_des_fea, s_qry_fea, s_tit_des_msk, s_itm_msk, s_tit_tok, s_tit_msk, s_des_tok, s_des_msk, s_qry_tok, s_qry_msk, y

    def __len__(self):
        return len(self.x)

class BaseModel(nn.Module):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None,
                 tokenizer=None, title_npy_folder="", query_npy_folder="", des_npy_folder="", userid2ad_list={}):

        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns

        self.tokenizer = tokenizer
        self.title_npy_folder = title_npy_folder
        self.query_npy_folder = query_npy_folder
        self.des_npy_folder = des_npy_folder
        self.userid2ad_list = userid2ad_list

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)
        #         nn.ModuleDict(
        #             {feat.embedding_name: nn.Embedding(feat.dimension, embedding_size, sparse=True) for feat in
        #              self.dnn_feature_columns}
        #         )

        self.linear_model = Linear(
            linear_feature_columns, self.feature_index, device=device)

        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        self.out = PredictionLayer(task, )
        self.to(device)

        # parameters for callbacks
        self._is_graph_network = True  # used for ModelCheckpoint in tf2
        self._ckpt_saved_epoch = False  # used for EarlyStopping in tf1.14
        self.history = History()

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None, test_x=None, test_y=None):
        """

        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        add_name_list = ["C6_ori", "C8_ori", "C9_ori", "C10_ori", "C2_ori"]
        #print("input xxx", x)
        if isinstance(x, dict):
            ori_x = [x[feature] for feature in add_name_list]
            x = [x[feature] for feature in list(self.feature_index.keys())]

        #print("XXX", x)
        #print("Ori xxx", ori_x)

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x_dict, val_y = validation_data
                ori_val_x = [val_x_dict[feature] for feature in add_name_list]
                val_x = [val_x_dict[feature] for feature in list(self.feature_index.keys())]
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
            ori_x, ori_val_x = (slice_arrays(ori_x, 0, split_at),
                                slice_arrays(ori_x, split_at))
            #print("split for ++++", ori_x)
            #print("split val ----", ori_val_x)

        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            #if i >= len(self.feature_index):
            #    x[i] = np.array(x[i].to_numpy().tolist())
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
            #print('+++', i, x[i], x[i].shape)

        for i in range(len(ori_x)):
            #ori_x[i] = np.array(ori_x[i].to_numpy().tolist())
            if len(ori_x[i].shape) == 1:
                ori_x[i] = np.expand_dims(ori_x[i], axis=1)
            #print("ori *****", i, ori_x[i], ori_x[i].shape)

        xxx = torch.from_numpy(
                np.concatenate(x, axis=-1))
        ori_xxx = torch.from_numpy(
                np.concatenate(ori_x, axis=-1))
        print("XXX shape:", xxx.shape, ori_xxx.shape, torch.from_numpy(y).shape)
        # train_tensor_data = Data.TensorDataset(
        #     xxx,
        #     torch.from_numpy(y))
        train_tensor_data = BaseDataset(xxx, ori_xxx, torch.from_numpy(y), self.title_npy_folder, self.query_npy_folder, self.des_npy_folder, self.userid2ad_list)

        # train_tensor_data = BaseDataset(xxx, torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size, num_workers=4)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            start = time.time()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    train_step_idx = 0
                    for _, (x_train, tit_fea, qry_fea, des_fea, tit_tok_fea, tit_msk_fea, qry_tok_fea, qry_msk_fea, des_tok_fea, des_msk_fea, s_tit_fea, s_des_fea, s_qry_fea, s_tit_des_msk, s_itm_msk, s_tit_tok, s_tit_msk, s_des_tok, s_des_msk, s_qry_tok, s_qry_msk, y_train) in t:
                        data_time.update(time.time() - start)
                        train_step_idx += 1
                        if train_step_idx % 20 == 0:
                            print('process', train_step_idx, len(train_loader))
                            print("Epoch:{}-{}/{}, loss: [{}], [{}], [{}] ".\
                              format(epoch, train_step_idx, len(train_loader), meter_to_str("Loss", losses, 6),
                                 meter_to_str("Batch_Time", batch_time, 6),
                                 meter_to_str("Data_Load_Time", data_time, 6)))
                        #start_time = time.time()
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        tit_fea = tit_fea.to(self.device)
                        qry_fea = qry_fea.to(self.device)
                        des_fea = des_fea.to(self.device)
                        tit_tok_fea = tit_tok_fea.to(self.device)
                        tit_msk_fea = tit_msk_fea.to(self.device)
                        qry_tok_fea = qry_tok_fea.to(self.device)
                        qry_msk_fea = qry_msk_fea.to(self.device)
                        des_tok_fea = des_tok_fea.to(self.device)
                        des_msk_fea = des_msk_fea.to(self.device)
                        s_tit_fea = s_tit_fea.to(self.device)
                        s_des_fea = s_des_fea.to(self.device)
                        s_qry_fea = s_qry_fea.to(self.device)
                        s_tit_des_msk = s_tit_des_msk.to(self.device)
                        s_itm_msk = s_itm_msk.to(self.device)
                        s_tit_tok = s_tit_tok.to(self.device)
                        s_tit_msk = s_tit_msk.to(self.device)
                        s_des_tok = s_des_tok.to(self.device)
                        s_des_msk = s_des_msk.to(self.device)
                        s_qry_tok = s_qry_tok.to(self.device)
                        s_qry_msk = s_qry_msk.to(self.device)
                        #print("Data input", time.time()-start_time)
                        #start_time = time.time()
                        #print("session fea", s_tit_fea.shape, s_des_fea.shape)
                        #print("tit_des msk", s_tit_des_msk)
                        #print("item msk", s_itm_msk)

                        y_pred = model(x, tit_fea, qry_fea, des_fea, tit_tok_fea, tit_msk_fea, qry_tok_fea, qry_msk_fea, des_tok_fea, des_msk_fea, s_tit_fea, s_des_fea, s_qry_fea, s_tit_des_msk, s_itm_msk, s_tit_tok, s_tit_msk, s_des_tok, s_des_msk, s_qry_tok, s_qry_msk).squeeze()
                        #print("forward time", time.time()-start_time)

                        #start_time = time.time()
                        optim.zero_grad()
                        if isinstance(loss_func, list):
                            assert len(loss_func) == self.num_tasks,\
                                "the length of `loss_func` should be equal with `self.num_tasks`"
                            loss = sum(
                                [loss_func[i](y_pred[:, i], y[:, i], reduction='sum') for i in range(self.num_tasks)])
                        else:
                            loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        #print("loss time", time.time()-start_time)
                        #start_time = time.time()
                        reg_loss = self.get_regularization_loss()
                        #print("reg loss time", time.time()-start_time)

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        #start_time = time.time()
                        total_loss.backward()
                        optim.step()
                        #print("optim time", time.time()-start_time)

                        losses.update(total_loss.item())
                        batch_time.update(time.time() - start)
                        start = time.time()

                        #start_time = time.time()
                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                try:
                                    train_result[name].append(metric_fun(
                                        y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))
                                except: continue
                        #print("metrics time", time.time()-start_time)


            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, ori_val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        try:
                            eval_str += " - " + "val_" + name + \
                                        ": {0: .4f}".format(epoch_logs["val_" + name])
                        except:
                            pass
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history

    def evaluate(self, x, ori_x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans, _ = self.predict(x, ori_x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            try:
                eval_result[name] = metric_fun(y, pred_ans)
            except Exception as err:
                print("Error in metric_fun", name, err)
                pass
        return eval_result

    def predict(self, x, ori_x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            #add_name_list = ["C6_ori", "C8_ori", "C9_ori"]
            add_name_list = ["C6_ori", "C8_ori", "C9_ori", "C10_ori", "C2_ori"]
            ori_x = [x[feature] for feature in add_name_list]
            x = [x[feature] for feature in list(self.feature_index.keys())]
            # x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            #if i >= len(self.feature_index):
            #    x[i] = np.array(x[i].tolist())
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        for i in range(len(ori_x)):
            #ori_x[i] = np.array(ori_x[i].to_numpy().tolist())
            if len(ori_x[i].shape) == 1:
                ori_x[i] = np.expand_dims(ori_x[i], axis=1)
            #print("ori *****", i, ori_x[i], ori_x[i].shape)

        # tensor_data = Data.TensorDataset(
        #     torch.from_numpy(np.concatenate(x, axis=-1)))
            # train_tensor_data = Data.TensorDataset(
            #     xxx,
            #     torch.from_numpy(y))
        xxx = torch.from_numpy(np.concatenate(x, axis=-1))
        ori_xxx = torch.from_numpy(
                np.concatenate(ori_x, axis=-1))
        y = np.ones(len(xxx))
        tensor_data = BaseDataset(xxx, ori_xxx, torch.from_numpy(y), self.title_npy_folder, self.query_npy_folder, self.des_npy_folder, self.userid2ad_list)

        print("Ori_xxx", ori_xxx.shape)
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size, num_workers=4)

        pred_ans = []
        pred_ori_ans = []
        sample_idx = 0
        with torch.no_grad():
            # NOTE: y_test is fake data
            for _, (x_test, tit_fea, qry_fea, des_fea, tit_tok_fea, tit_msk_fea, qry_tok_fea, qry_msk_fea, des_tok_fea, des_msk_fea, s_tit_fea, s_des_fea, s_qry_fea, s_tit_des_msk, s_itm_msk, s_tit_tok, s_tit_msk, s_des_tok, s_des_msk, s_qry_tok, s_qry_msk, y_test) in enumerate(test_loader):
                x = x_test.to(self.device).float()  # x_test[0]???
                tit_fea = tit_fea.to(self.device)
                qry_fea = qry_fea.to(self.device)
                des_fea = des_fea.to(self.device)
                tit_tok_fea = tit_tok_fea.to(self.device)
                tit_msk_fea = tit_msk_fea.to(self.device)
                qry_tok_fea = qry_tok_fea.to(self.device)
                qry_msk_fea = qry_msk_fea.to(self.device)
                des_tok_fea = des_tok_fea.to(self.device)
                des_msk_fea = des_msk_fea.to(self.device)
                s_tit_fea = s_tit_fea.to(self.device)
                s_des_fea = s_des_fea.to(self.device)
                s_qry_fea = s_qry_fea.to(self.device)
                s_tit_des_msk = s_tit_des_msk.to(self.device)
                s_itm_msk = s_itm_msk.to(self.device)
                s_tit_tok = s_tit_tok.to(self.device)
                s_tit_msk = s_tit_msk.to(self.device)
                s_des_tok = s_des_tok.to(self.device)
                s_des_msk = s_des_msk.to(self.device)
                s_qry_tok = s_qry_tok.to(self.device)
                s_qry_msk = s_qry_msk.to(self.device)

                y_pred = model(x, tit_fea, qry_fea, des_fea, tit_tok_fea, tit_msk_fea, qry_tok_fea, qry_msk_fea, des_tok_fea, des_msk_fea, s_tit_fea, s_des_fea, s_qry_fea, s_tit_des_msk, s_itm_msk, s_tit_tok, s_tit_msk, s_des_tok, s_des_msk, s_qry_tok, s_qry_msk).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)
                for idx in range(len(y_pred)):
                    meta_info_list = ori_xxx[sample_idx]
                    userid = int(meta_info_list[3])
                    adid = int(meta_info_list[4])
                    queryid = int(meta_info_list[0])
                    pred_ori_ans.append([userid, adid, queryid])
                    sample_idx += 1

        return np.concatenate(pred_ans).astype("float64"), pred_ori_ans

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer,
                loss=None,
                metrics=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=0.01)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters())  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            loss_func = self._get_loss_func_single(loss)
        elif isinstance(loss, list):
            loss_func = [self._get_loss_func_single(loss_single) for loss_single in loss]
        else:
            loss_func = loss
        return loss_func

    def _get_loss_func_single(self, loss):
        if loss == "binary_crossentropy":
            loss_func = F.binary_cross_entropy
        elif loss == "mse":
            loss_func = F.mse_loss
        elif loss == "mae":
            loss_func = F.l1_loss
        else:
            raise NotImplementedError
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    @staticmethod
    def _accuracy_score(y_true, y_pred):
        return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = self._accuracy_score
                self.metrics_names.append(metric)
        return metrics_

    def _in_multi_worker_mode(self):
        # used for EarlyStopping in tf1.15
        return None

    @property
    def embedding_size(self, ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]
