# -*- coding:utf-8 -*-
"""
Author:
    Wutong Zhang
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""
import json
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from .basemodel_bert import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN, CIN


class xDeepFM(BaseModel):
    """Instantiates the xDeepFM architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param cin_layer_size: list,list of positive integer or empty list, the feature maps  in each hidden layer of Compressed Interaction Network
    :param cin_split_half: bool.if set to True, half of the feature maps in each hidden will connect to output unit
    :param cin_activation: activation function used on feature maps
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: L2 regularizer strength applied to deep net
    :param l2_reg_cin: L2 regularizer strength applied to CIN.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 256),
                 cin_layer_size=(256, 128,), cin_split_half=True, cin_activation='relu', l2_reg_linear=0.00001,
                 l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, init_std=0.0001, seed=1024, dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None,
                 nlp_dim=32, word_size=31000, word_dim=32, score_size=101, score_dim=32, tokenizer=None, title_npy_folder="", query_npy_folder="", des_npy_folder="", userid2ad_list={}):

        super(xDeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus, tokenizer=tokenizer, title_npy_folder=title_npy_folder,
                                     query_npy_folder=query_npy_folder, des_npy_folder=des_npy_folder, userid2ad_list=userid2ad_list)
        self.dnn_hidden_units = dnn_hidden_units
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)

            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.cin_layer_size = cin_layer_size
        self.use_cin = len(self.cin_layer_size) > 0 and len(dnn_feature_columns) > 0
        if self.use_cin:
            field_num = len(self.embedding_dict)
            if cin_split_half == True:
                self.featuremap_num = sum(
                    cin_layer_size[:-1]) // 2 + cin_layer_size[-1]
            else:
                self.featuremap_num = sum(cin_layer_size)
            self.cin = CIN(field_num, cin_layer_size,
                           cin_activation, cin_split_half, l2_reg_cin, seed, device=device)
            self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False).to(device)
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0], self.cin.named_parameters()),
                                           l2=l2_reg_cin)

        multi_bert_config = BertConfig(vocab_size=1, max_position_embeddings=1, hidden_size=768, num_hidden_layers=4, num_attention_heads=4)
        self.multi_transformer_bro = BertModel(multi_bert_config).encoder
        multi_bert_token_config = BertConfig(vocab_size=1, max_position_embeddings=1, hidden_size=32, num_hidden_layers=4, num_attention_heads=4, intermediate_size=128)
        self.multi_transformer_tok = BertModel(multi_bert_token_config).encoder

        self.word_embedding = nn.Embedding(word_size, word_dim)
        nn.init.xavier_uniform_(self.word_embedding.weight)
        #self.user_position_embedding = nn.Embedding(100, 768)
        self.nlp_linear = nn.Linear(nlp_dim*4, 1, bias=False).to(device)
        self.add_regularization_weight(self.nlp_linear.weight, l2=l2_reg_dnn)
        self.nlp_token_linear = nn.Linear(word_dim*4, 1, bias=False).to(device)
        self.add_regularization_weight(self.nlp_token_linear.weight, l2=l2_reg_dnn)
        #----start add BERT----
        #self.bert = bert_model

        self.title_fc = nn.Sequential(nn.Linear(768, 256), nn.ReLU(),
                                      nn.Dropout(p=0.2), nn.Linear(256, nlp_dim))
        self.session_reduce_fc = nn.Sequential(nn.Linear(768, 256), nn.ReLU(),
                                      nn.Dropout(p=0.2), nn.Linear(256, nlp_dim))

        self.title_des_fc = nn.Sequential(nn.Linear(768, 1), nn.Tanh())
        nn.init.xavier_uniform_(self.title_des_fc[0].weight)
        self.session_fc = nn.Sequential(nn.Linear(768, 1), nn.Tanh())
        nn.init.xavier_uniform_(self.session_fc[0].weight)
        self.title_des_token_fc = nn.Sequential(nn.Linear(word_dim, 1), nn.Tanh())
        nn.init.xavier_uniform_(self.title_des_token_fc[0].weight)
        self.session_token_fc = nn.Sequential(nn.Linear(word_dim, 1), nn.Tanh())
        nn.init.xavier_uniform_(self.session_token_fc[0].weight)

        self.tit_score_embedding = nn.Embedding(score_size, score_dim)
        self.des_score_embedding = nn.Embedding(score_size, score_dim)
        self.qry_score_embedding = nn.Embedding(score_size, score_dim)
        nn.init.xavier_uniform_(self.tit_score_embedding.weight)
        nn.init.xavier_uniform_(self.des_score_embedding.weight)
        nn.init.xavier_uniform_(self.qry_score_embedding.weight)
        self.nlp_token_score_linear = nn.Linear(score_dim*3, 1, bias=False).to(device)
        self.add_regularization_weight(self.nlp_token_score_linear.weight, l2=l2_reg_dnn)

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

    def add_pos_emb(self, bs, his_emb, start_pos=0):
        # his_emb: bs, 30, 768
        his_bro_long = torch.arange(start_pos, start_pos+his_emb.shape[1]).repeat(bs, 1).to(
            self.device)  # bs, 30 --> [[0,1,2,...],[0,1,2,...],...]
        his_bro_pos = self.user_position_embedding(his_bro_long)  # bs, 30, 768
        return his_bro_pos + his_emb  # bs, 30, 768

    def atten_title_des(self, bs, s_tit_fea, s_des_fea, s_qry_fea, s_tit_des_msk, title_des_fc):
        s_tit_att_fea = s_tit_fea.reshape(-1, s_tit_fea.shape[-1])  # (bs*10), 768
        s_des_att_fea = s_des_fea.reshape(-1, s_des_fea.shape[-1])  # (bs*10), 768
        s_qry_att_fea = s_qry_fea.reshape(-1, s_qry_fea.shape[-1])  # (bs*10), 768
        s_att_msk = s_tit_des_msk.reshape(-1, s_tit_des_msk.shape[-1])  # (bs*10), 3

        s_itm_fea = torch.stack([s_tit_att_fea, s_des_att_fea, s_qry_att_fea], dim=1)  # (bs*10), 3, 768
        s_att_feat = self.attention_by_weight(s_itm_fea, s_att_msk, title_des_fc)  # (bs*10), 768
        s_att_feat = s_att_feat.reshape(bs, -1, s_att_feat.shape[-1])  # bs, 10, 768
        return s_att_feat

    def _get_nlp_feat(self, word_id, word_msk):
        # word_id/word_msk: bs, #words
        #print("ids", word_id.shape, word_id)
        #print("msk", word_msk.shape, word_msk)
        word_fea = self.word_embedding(word_id)  # bs, #words, 32
        return torch.sum(word_fea, dim=1)  # bs, 32

    def get_multi_his_feat(self, his_feat, his_mask, multi_tran):
        """
        param: his_feat: bs, 30, 768
        param: his_mask, bs, 30
        """
        # multi_attention_mask = his_mask.unsqueeze(1).unsqueeze(2)
        # multi_attention_mask = multi_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # multi_attention_mask = (1.0 - multi_attention_mask) * -10000.0
        # multi_attention_mask = multi_attention_mask.to(self.device)
        his_mask_2 = his_mask.unsqueeze(1).unsqueeze(2)
        his_mask_2 = his_mask_2.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        his_mask_2 = (1.0 - his_mask_2) * -10000.0
        his_mask_2 = his_mask_2.to(self.device)
        #print("mask", his_mask_2.shape, his_mask_2)
        #print("mask", self.check_nan(his_mask_2))

        #print("input_feat", his_feat.shape, his_feat)
        #print("input_feat", self.check_nan(his_feat))
        multi_encoded_out = multi_tran(his_feat, his_mask_2, return_dict=True).last_hidden_state
        #print("multi_out", multi_encoded_out.shape, multi_encoded_out)
        #print("multi_out", self.check_nan(multi_encoded_out))
        # print("multi_encoded_out", multi_encoded_out.shape)
        his_atten_feat = torch.mean(multi_encoded_out, dim=1)  # bs, 768
        return his_atten_feat

    def check_nan(self, a):
        #print(a.shape)
        return torch.isnan(a).any()

    def forward(self, X_linear_feat, tit_feat, qry_feat, des_feat, tit_tok_fea, tit_msk_fea, qry_tok_fea, qry_msk_fea, des_tok_fea, des_msk_fea, s_tit_fea, s_des_fea, s_qry_fea, s_tit_des_msk, s_itm_msk, s_tit_tok, s_tit_msk, s_des_tok, s_des_msk, s_qry_tok, s_qry_msk, tit_tok_score, des_tok_score, qry_tok_score):
        bs = tit_feat.shape[0]
        #print("BS", bs)
        tit_feat = self.title_fc(tit_feat)
        qry_feat = self.title_fc(qry_feat)
        des_feat = self.title_fc(des_feat)
        #item_feat = torch.stack([tit_feat, des_feat, qry_feat], dim=1)  # bs, 3, 32
        #item_feat = torch.mean(item_feat, dim=1)  # bs, 32

        tit_token_feat = self._get_nlp_feat(tit_tok_fea, tit_msk_fea)  # bs, 32
        qry_token_feat = self._get_nlp_feat(qry_tok_fea, qry_msk_fea)  # bs, 32
        des_token_feat = self._get_nlp_feat(des_tok_fea, des_msk_fea)  # bs, 32
        #item_token_feat = torch.stack([tit_token_feat, qry_token_feat, des_token_feat], dim=1)  # bs, 3, 32
        #item_token_feat = torch.mean(item_token_feat, dim=1)  # bs, 32

        #print("---> ", s_tit_fea.shape, s_des_fea.shape, s_qry_fea.shape, s_tit_des_msk.shape)
        #s_tit_fea = self.add_pos_emb(s_tit_fea.shape[0], s_tit_fea, 0)
        #s_des_fea = self.add_pos_emb(s_des_fea.shape[0], s_des_fea, 30)
        #s_qry_fea = self.add_pos_emb(s_qry_fea.shape[0], s_qry_fea, 60)

        s_tit_token_feat = self._get_nlp_feat(s_tit_tok.reshape(-1, s_tit_tok.shape[-1]), s_tit_msk.reshape(-1, s_tit_msk.shape[-1]))  # (bs*#items), 32
        s_tit_token_feat = s_tit_token_feat.reshape(bs, -1, s_tit_token_feat.shape[-1])  # bs, #items, 32
        s_des_token_feat = self._get_nlp_feat(s_des_tok.reshape(-1, s_des_tok.shape[-1]), s_des_msk.reshape(-1, s_des_msk.shape[-1]))
        s_des_token_feat = s_des_token_feat.reshape(bs, -1, s_des_token_feat.shape[-1])  # bs, #items, 32
        s_qry_token_feat = self._get_nlp_feat(s_qry_tok.reshape(-1, s_qry_tok.shape[-1]), s_qry_msk.reshape(-1, s_qry_msk.shape[-1]))
        s_qry_token_feat = s_qry_token_feat.reshape(bs, -1, s_qry_token_feat.shape[-1])  # bs, #items, 32

        s_itm_feat = self.atten_title_des(s_tit_fea.shape[0], s_tit_fea, s_des_fea, s_qry_fea, s_tit_des_msk, self.title_des_fc)  # bs, 10, 768
        #s_itm_feat = torch.stack([s_tit_fea, s_des_fea, s_qry_fea], dim=1)  # bs, 3, 10, 768
        #s_itm_feat = torch.mean(s_itm_feat, dim=1)  # bs, 10, 768

        s_itm_token_feat = self.atten_title_des(s_tit_token_feat.shape[0], s_tit_token_feat, s_des_token_feat, s_qry_token_feat, s_tit_des_msk, self.title_des_token_fc)  # bs, 10, 32
        #s_itm_token_feat = torch.stack([s_tit_token_feat, s_des_token_feat, s_qry_token_feat], dim=1)  # bs, 3, 10, 32
        #s_itm_token_feat = torch.mean(s_itm_token_feat, dim=1)  # bs, 10, 32

        #print("SSS", s_itm_feat.shape)
        #s_att_feat = self.attention_by_weight(s_itm_feat, s_itm_msk, self.session_fc)  # bs, 768
        #s_att_feat = torch.mean(s_itm_feat, dim=1)  # bs, 768
        s_att_feat = self.get_multi_his_feat(s_itm_feat, s_itm_msk, self.multi_transformer_bro)  # bs, 768

        s_att_feat = self.session_reduce_fc(s_att_feat)  # bs, 32

        #s_att_token_feat = self.attention_by_weight(s_itm_token_feat, s_itm_msk, self.session_token_fc)  # bs, 32
        #s_att_token_feat = torch.mean(s_itm_token_feat, dim=1)  # bs, 32
        s_att_token_feat = self.get_multi_his_feat(s_itm_token_feat, s_itm_msk, self.multi_transformer_tok)  # bs, 32
        #print('s_att_feat', s_att_feat.shape)

        nlp_bert_feat = torch.cat([tit_feat, des_feat, qry_feat, s_att_feat], dim=-1)   # bs, (32*4)
        nlp_bert_token_feat = torch.cat([tit_token_feat, des_token_feat, qry_token_feat, s_att_token_feat], dim=-1)  # bs, (32*4)

        # get score embedding
        tit_score_emb = self.tit_score_embedding(tit_tok_score).squeeze(1)  # bs, 32
        des_score_emb = self.des_score_embedding(des_tok_score).squeeze(1)  # bs, 32
        qry_score_emb = self.qry_score_embedding(qry_tok_score).squeeze(1)  # bs, 32
        #print("score emb", tit_score_emb.shape, des_score_emb.shape, qry_score_emb.shape)
        tok_score_emb = torch.cat([tit_score_emb, des_score_emb, qry_score_emb], dim=-1)  # bs, (32*3)

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X_linear_feat, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        linear_logit = self.linear_model(X_linear_feat)
        if self.use_cin:
            cin_input = torch.cat(sparse_embedding_list, dim=1)
            cin_output = self.cin(cin_input)
            cin_logit = self.cin_linear(cin_output)
        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)

        if len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) == 0:  # only linear
            final_logit = linear_logit
        elif len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) > 0:  # linear + CIN
            final_logit = linear_logit + cin_logit
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) == 0:  # linear +　Deep
            final_logit = linear_logit + dnn_logit
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) > 0:  # linear + CIN + Deep
            final_logit = linear_logit + dnn_logit + cin_logit
        else:
            raise NotImplementedError

        final_logit += self.nlp_linear(nlp_bert_feat)
        final_logit += self.nlp_token_linear(nlp_bert_token_feat)
        final_logit += self.nlp_token_score_linear(tok_score_emb)
        y_pred = self.out(final_logit)

        return y_pred
