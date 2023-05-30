import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import BertModel
from transformers.activations import GELUActivation
from transformers.models.bert.modeling_bert import BertPreTrainingHeads


class NLPBertModel(nn.Module):
    def __init__(self, bert_model, device, model_flag="bert",
                 feat_hidden_size=768, hidden_size=128, dropout_prob=0.1):
        super(NLPBertModel, self).__init__()

        self.model_flag = model_flag
        self.bert = bert_model
        print('config', self.bert.config)
        if model_flag == "bert":
            self.cls = BertPreTrainingHeads(self.bert.config)  # , self.bert.embeddings.word_embeddings.weight
            self.cls.predictions.decoder.weight = self.bert.embeddings.word_embeddings.weight
        elif model_flag == "distilbert":
            print("using distil bert =======")
            self.activation = GELUActivation()
            self.vocab_transform = nn.Linear(self.bert.config.dim, self.bert.config.dim)
            self.vocab_layer_norm = nn.LayerNorm(self.bert.config.dim, eps=1e-12)
            self.vocab_projector = nn.Linear(self.bert.config.dim, self.bert.config.vocab_size)
            #print("ssss", self.bert.embeddings.word_embeddings.weight.shape)
            self.vocab_projector.weight = self.bert.embeddings.word_embeddings.weight
        else:
            raise NotImplementedError
        self.device = device
        self.cur_fc = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_bro_fc = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_add_fc = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_buy_fc = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_bro_fc_item = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_add_fc_item = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_buy_fc_item = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_qry_fc_item = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())
        self.his_fc = nn.Sequential(nn.Linear(feat_hidden_size, 1), nn.Tanh())

        self.item_fc = nn.Sequential(nn.Linear(feat_hidden_size, hidden_size))
        self.user_fc = nn.Sequential(nn.Linear(feat_hidden_size, hidden_size))

        self.final_fc = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, 1)
            )

        nn.init.xavier_uniform_(self.cur_fc[0].weight)
        nn.init.xavier_uniform_(self.his_bro_fc[0].weight)
        nn.init.xavier_uniform_(self.his_add_fc[0].weight)
        nn.init.xavier_uniform_(self.his_buy_fc[0].weight)
        nn.init.xavier_uniform_(self.his_bro_fc_item[0].weight)
        nn.init.xavier_uniform_(self.his_add_fc_item[0].weight)
        nn.init.xavier_uniform_(self.his_buy_fc_item[0].weight)
        nn.init.xavier_uniform_(self.his_qry_fc_item[0].weight)
        nn.init.xavier_uniform_(self.his_fc[0].weight)
        nn.init.xavier_uniform_(self.item_fc[0].weight)
        nn.init.xavier_uniform_(self.user_fc[0].weight)
        nn.init.xavier_uniform_(self.final_fc[0].weight)
        nn.init.xavier_uniform_(self.final_fc[-1].weight)

    def print_model_param(self):
        print("cur_fc", torch.max(self.cur_fc[0].weight), torch.min(self.cur_fc[0].weight))
        print("his_bro_fc", torch.max(self.his_bro_fc[0].weight), torch.min(self.his_bro_fc[0].weight))
        print("his_add_fc", torch.max(self.his_add_fc[0].weight), torch.min(self.his_add_fc[0].weight))
        print("his_buy_fc", torch.max(self.his_buy_fc[0].weight), torch.min(self.his_buy_fc[0].weight))
        print("his_bro_fc_item", torch.max(self.his_bro_fc_item[0].weight), torch.min(self.his_bro_fc_item[0].weight))
        print("his_add_fc_item", torch.max(self.his_add_fc_item[0].weight), torch.min(self.his_add_fc_item[0].weight))
        print("his_buy_fc_item", torch.max(self.his_buy_fc_item[0].weight), torch.min(self.his_buy_fc_item[0].weight))
        print("his_fc", torch.max(self.his_fc[0].weight), torch.min(self.his_fc[0].weight))
        print("item_fc", torch.max(self.item_fc[0].weight), torch.min(self.item_fc[0].weight))
        print("user_fc", torch.max(self.user_fc[0].weight), torch.min(self.user_fc[0].weight))
        print("final_fc", torch.max(self.final_fc[0].weight), torch.min(self.final_fc[0].weight))

    def get_atten_score_for_item_old(self, cur_id_tensor, cur_mask_tensor):
        """
        get attention scores: bs, #tokens
        """
        token_prob, cur_att_feat = self.get_bert_token_score(cur_id_tensor, cur_mask_tensor, self.cur_fc)  # bs, #tokens
        itm_feat = self.item_fc(cur_att_feat)  # bs, 128
        return token_prob, itm_feat

    def get_atten_score_for_item(self, cur_id_tensor, cur_mask_tensor):
        token_prob, cur_att_feat = self.get_bert_cls_atten_feat(cur_id_tensor, cur_mask_tensor)  # bs, #tokens, 768
        #print("token_prob", token_prob)
        itm_feat = self.item_fc(cur_att_feat)  # bs, 128
        return token_prob, itm_feat

    def get_bert_feat_for_item(self, cur_id_tensor, cur_mask_tensor, is_title=True):
        token_prob, cur_att_feat = self.get_bert_cls_atten_feat(cur_id_tensor, cur_mask_tensor)  # bs, #tokens, 768
        return token_prob, cur_att_feat

    def norm_matrix(self, emb, dim=1):
        """
        特征归一化
        :param emb: 输入的特征，bs * dim
        :param dim: 按行或列进行归一化
        :return:
        """
        emb_norm = emb.norm(p=2, dim=dim, keepdim=True)
        return emb.div(emb_norm)

    def get_atten_score_for_user(self, his_msk_tensor, his_bro_msk_tensor, his_add_msk_tensor, his_buy_msk_tensor, \
                his_bro_fea_tensor, his_add_fea_tensor, his_buy_fea_tensor, query_id_tensor, query_mask_tensor):
        bs = his_msk_tensor.shape[0]
        his_bro_att_feat = his_bro_fea_tensor
        his_add_att_feat = his_add_fea_tensor
        his_buy_att_feat = his_buy_fea_tensor

        his_bro_att_item_feat = self.attention_by_weight(his_bro_att_feat, his_bro_msk_tensor,
                                                         self.his_bro_fc_item)  # bs, 768
        his_add_att_item_feat = self.attention_by_weight(his_add_att_feat, his_add_msk_tensor,
                                                         self.his_add_fc_item)  # bs, 768
        his_buy_att_item_feat = self.attention_by_weight(his_buy_att_feat, his_buy_msk_tensor,
                                                         self.his_buy_fc_item)  # bs, 768
        bert_feat = self.bert(query_id_tensor, query_mask_tensor)  # bs, #tokens, 768
        bert_feat = bert_feat.last_hidden_state
        cls_feat = bert_feat[:, 0, :].unsqueeze(-1)  # bs, 768, 1
        bert_feat_norm = self.norm_matrix(bert_feat, dim=-1)
        cls_feat_norm = self.norm_matrix(cls_feat, dim=1)
        atten_score = torch.bmm(bert_feat_norm, cls_feat_norm).squeeze(-1)  # bs, #tokens
        # print("score", atten_score)
        extended_attention_mask = (1.0 - query_mask_tensor) * -10000.0  # 1-->0, 0-->-inf
        atten_score = atten_score + extended_attention_mask
        query_probs = nn.Softmax(dim=-1)(atten_score)

        user_stack_feat = torch.stack([his_bro_att_item_feat, his_add_att_item_feat, his_buy_att_item_feat, bert_feat[:, 0, :]],
                                      dim=1)  # bs, 4, 768
        his_prob, usr_att_feat = self.attention_score_by_weight(user_stack_feat, his_msk_tensor, self.his_fc)
        usr_feat = self.user_fc(usr_att_feat)  # bs, 128

        return query_probs, his_prob, usr_feat

    def forward_lm(self, cur_id_tensor, cur_mask_tensor, masked_pos_tensor):
        # segment_ids = torch.zeros_like(cur_mask_tensor).long()
        bert_feat = self.bert(cur_id_tensor, cur_mask_tensor)
        bert_feat = bert_feat.last_hidden_state
        mask_feature = torch.gather(bert_feat, 1,
                                    masked_pos_tensor.unsqueeze(2).expand(-1, -1, bert_feat.shape[-1]))
        #print("mask_feature", bert_feat.shape, mask_feature.shape, masked_pos_tensor.shape)
        if self.model_flag == "bert":
            prediction_scores, _ = self.cls(mask_feature, bert_feat)
        else:
            prediction_logits = self.vocab_transform(mask_feature)  # (bs, seq_length, dim)
            prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_scores = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
            #print('score', prediction_scores.shape)
        #print("score", prediction_scores.shape)
        return prediction_scores

    def forward(self, cur_id_tensor, cur_mask_tensor, is_title=True):
        token_prob, cur_att_feat = self.get_bert_cls_atten_feat(cur_id_tensor, cur_mask_tensor)  # bs, #tokens, 768
        return token_prob, cur_att_feat

    def get_bert_feat(self, ids, mask, weight):
        segment_ids = torch.zeros_like(mask).long()
        bert_feat = self.bert(ids, mask, segment_ids)
        bert_feat = bert_feat.last_hidden_state
        atten_feat = self.attention_by_weight(bert_feat, mask, weight)
        return atten_feat

    def get_bert_cls_feat(self, ids, mask):
        # segment_ids = torch.zeros_like(mask).long()
        bert_feat = self.bert(ids, mask)  # segment_ids
        bert_feat = bert_feat.last_hidden_state
        return bert_feat[:, 0, :]

    def get_bert_cls_atten_feat(self, ids, mask):
        bert_feat = self.bert(ids, mask)  # segment_ids
        bert_feat = bert_feat.last_hidden_state  # bs, #tokens, 768
        cls_feat = bert_feat[:, 0, :].unsqueeze(-1)  # bs, 768, 1
        bert_feat_norm = self.norm_matrix(bert_feat, dim=-1)
        cls_feat_norm = self.norm_matrix(cls_feat, dim=1)

        atten_score = torch.bmm(bert_feat_norm, cls_feat_norm).squeeze(-1)  # bs, #tokens
        # print("score", atten_score)
        extended_attention_mask = (1.0 - mask) * -10000.0  # 1-->0, 0-->-inf
        atten_score = atten_score + extended_attention_mask
        atten_probs = nn.Softmax(dim=-1)(atten_score)
        # print("mask", mask)
        # print("probs", atten_probs)
        atten_feat = atten_probs.unsqueeze(-1) * bert_feat
        atten_feat = torch.sum(atten_feat, dim=1)
        return atten_probs, atten_feat

    def get_bert_token_score(self, ids, mask, weight):
        segment_ids = torch.zeros_like(mask).long()
        bert_feat = self.bert(ids, mask, segment_ids)
        bert_feat = bert_feat.last_hidden_state
        atten_prob, atten_feat = self.attention_score_by_weight(bert_feat, mask, weight)
        return atten_prob, atten_feat

    def attention_score_by_weight(self, feat, mask, weight):
        atten_score = weight(feat).squeeze(-1)  # bs, #tokens
        extended_attention_mask = (1.0 - mask) * -10000.0  # 1-->0, 0-->-inf
        atten_score = atten_score + extended_attention_mask
        atten_probs = nn.Softmax(dim=-1)(atten_score)  # bs, #tokens
        atten_feat = atten_probs.unsqueeze(-1) * feat
        atten_feat = torch.sum(atten_feat, dim=1)
        return atten_probs, atten_feat

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
        #print("atten_probs", atten_probs)
        atten_feat = atten_probs.unsqueeze(-1) * feat
        atten_feat = torch.sum(atten_feat, dim=1)
        return atten_feat
