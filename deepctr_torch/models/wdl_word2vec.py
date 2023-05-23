# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016: 7-10.(https://arxiv.org/pdf/1606.07792.pdf)
"""
import json
import torch
import torch.nn as nn

from .basemodel_word2vec import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN


class WDL(BaseModel):
    """Instantiates the Wide&Deep Learning architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 128),
                 l2_reg_linear=1e-5,
                 l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 dnn_use_bn=False,
                 task='binary', device='cpu', gpus=None,
                 nlp_dim=32, tokenizer=None, title_npy_folder="", query_npy_folder="", des_npy_folder="", userid2ad_list={}):

        super(WDL, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus, tokenizer=tokenizer, title_npy_folder=title_npy_folder,
                                     query_npy_folder=query_npy_folder, des_npy_folder=des_npy_folder, userid2ad_list=userid2ad_list)

        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.nlp_linear = nn.Linear(nlp_dim*4, 1, bias=False).to(device)
        self.add_regularization_weight(self.nlp_linear.weight, l2=l2_reg_dnn)
        #----start add BERT----
        #self.bert = bert_model

        self.title_fc = nn.Sequential(nn.Linear(200, 64), nn.ReLU(),
                                      nn.Dropout(p=0.2), nn.Linear(64, nlp_dim))
        self.session_reduce_fc = nn.Sequential(nn.Linear(200, 64), nn.ReLU(),
                                      nn.Dropout(p=0.2), nn.Linear(64, nlp_dim))

        self.title_des_fc = nn.Sequential(nn.Linear(200, 1), nn.Tanh())
        nn.init.xavier_uniform_(self.title_des_fc[0].weight)
        self.session_fc = nn.Sequential(nn.Linear(200, 1), nn.Tanh())
        nn.init.xavier_uniform_(self.session_fc[0].weight)

        #-----end-------
        self.device = device
        self.to(device)

    def norm_matrix(self, emb, dim=1):
        """
        特征归一化
        :param emb: 输入的特征，bs * dim
        :param dim: 按行或列进行归一化
        :return:
        """
        emb_norm = emb.norm(p=2, dim=dim, keepdim=True)
        return emb.div(emb_norm)

    # def get_bert_cls_atten_feat(self, ids, mask):
    #     bert_feat = self.bert(ids, mask)  # segment_ids
    #     bert_feat = bert_feat.last_hidden_state  # bs, #tokens, 768
    #     cls_feat = bert_feat[:, 0, :].unsqueeze(-1)  # bs, 768, 1
    #     bert_feat_norm = self.norm_matrix(bert_feat, dim=-1)
    #     cls_feat_norm = self.norm_matrix(cls_feat, dim=1)
    #
    #     atten_score = torch.bmm(bert_feat_norm, cls_feat_norm).squeeze(-1)  # bs, #tokens
    #     # print("score", atten_score)
    #     extended_attention_mask = (1.0 - mask) * -10000.0  # 1-->0, 0-->-inf
    #     atten_score = atten_score + extended_attention_mask
    #     atten_probs = nn.Softmax(dim=-1)(atten_score)
    #     # print("mask", mask)
    #     # print("probs", atten_probs)
    #     atten_feat = atten_probs.unsqueeze(-1) * bert_feat
    #     atten_feat = torch.sum(atten_feat, dim=1)  # bs, 768
    #     atten_feat = self.title_fc(atten_feat)     # bs, 32
    #     return atten_probs, atten_feat
    
    def attention_by_weight(self, feat, mask, weight):
        """
        feat: bs, n, 768
        mask: bs, n
        weight: 768
        """
        atten_score = weight(feat).squeeze(-1)  # bs, #tokens
        extended_attention_mask = (1.0 - mask) * -10000.0  # 1-->0, 0-->-inf
        atten_score = atten_score + extended_attention_mask
        atten_probs = nn.Softmax(dim=-1)(atten_score)
        #print("msk", mask)
        #print("atten_probs", atten_probs)
        atten_feat = atten_probs.unsqueeze(-1) * feat
        atten_feat = torch.sum(atten_feat, dim=1)
        return atten_feat

    def atten_title_des(self, bs, s_tit_fea, s_des_fea, s_qry_fea, s_tit_des_msk):
        s_tit_att_fea = s_tit_fea.reshape(-1, s_tit_fea.shape[-1])  # (bs*10), 768
        s_des_att_fea = s_des_fea.reshape(-1, s_des_fea.shape[-1])  # (bs*10), 768
        s_qry_att_fea = s_qry_fea.reshape(-1, s_qry_fea.shape[-1])  # (bs*10), 768
        s_att_msk = s_tit_des_msk.reshape(-1, s_tit_des_msk.shape[-1])  # (bs*10), 3

        s_itm_fea = torch.stack([s_tit_att_fea, s_des_att_fea, s_qry_att_fea], dim=1)  # (bs*10), 3, 768
        s_att_feat = self.attention_by_weight(s_itm_fea, s_att_msk, self.title_des_fc)  # (bs*10), 768
        s_att_feat = s_att_feat.reshape(bs, -1, s_att_feat.shape[-1])  # bs, 10, 768
        return s_att_feat

    def forward(self, X_linear_feat, tit_feat, qry_feat, des_feat, s_tit_fea, s_des_fea, s_qry_fea, s_tit_des_msk, s_itm_msk):
        tit_feat = self.title_fc(tit_feat)
        qry_feat = self.title_fc(qry_feat)
        des_feat = self.title_fc(des_feat)

        #print("---> ", s_tit_fea.shape, s_des_fea.shape, s_qry_fea.shape, s_tit_des_msk.shape)
        s_itm_feat = self.atten_title_des(s_tit_fea.shape[0], s_tit_fea, s_des_fea, s_qry_fea, s_tit_des_msk)  # bs, 10, 768
        #print("SSS", s_itm_feat.shape)        
        s_att_feat = self.attention_by_weight(s_itm_feat, s_itm_msk, self.session_fc)  # bs, 768
        s_att_feat = self.session_reduce_fc(s_att_feat)  # bs, 32 
        #print('s_att_feat', s_att_feat.shape)

        nlp_bert_feat = torch.cat([tit_feat, qry_feat, des_feat, s_att_feat], dim=-1)   # bs, (32*3)

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X_linear_feat, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        logit = self.linear_model(X_linear_feat)

        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        logit += self.nlp_linear(nlp_bert_feat)
        y_pred = self.out(logit)

        return y_pred
