import os, pickle
import json
import bisect
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from functools import partial

from utils import load_obj_from_zip

class NLPDataset(Dataset):
    def __init__(self, title_path, npy_folder, ctr_path, tokenizer, pickle_flag=False, his_browse_num=30, pkl_name_substr_len=3,
                 his_add_to_cart_num=10, his_buy_num=10, his_sim_num=150, max_token_num=512, dense_input_size=101, emb_hidden_size=128, debug=False):
        self.tokenizer = tokenizer
        self.npy_folder = npy_folder
        self.his_browse_num = his_browse_num
        self.his_add_to_cart_num = his_add_to_cart_num
        self.his_buy_num = his_buy_num
        self.max_token_num = max_token_num
        self.his_sim_num = his_sim_num
        self.emb_hidden_size = emb_hidden_size
        #self.itemid2title = self._load_title(title_path)
        self.dense_input_size = dense_input_size
        self.pickle_flag = pickle_flag
        pkl_name_substr_len = 0   ######TODO
        self.pkl_name_substr_len = pkl_name_substr_len  # merge multiple item features into 1 pkl file
        self.has_pkl_item_set = self._load_itemset()
        self.samples = self._load_ctr_offline_new(ctr_path, debug)
        self.stat_pos_and_neg()

    def _update_sample(self):
        self.samples = [v for v in self.samples if str(v[2]) in self.itemid2info]
        # for one_sample in self.samples:
        #     current_itemid = one_sample[2]
        #     if current_itemid not in self.itemid2info:
        #         self.samples.remove(one_sample)

    def _load_emb_info(self, emb_path):
        itemid2info = {}
        with open(emb_path, encoding='utf8') as f:
            for lidx, line in enumerate(f):
                if lidx % 100000 == 0:
                    print("load emb ...", lidx)
                tmp = line.strip('\n').split('\t')
                itemid = tmp[0]
                if itemid not in self.need_itemid_set: continue
                itemid2info[itemid] = json.loads(tmp[1])
        return itemid2info

    def stat_pos_and_neg(self):
        pos, neg = 0, 0
        for one in self.samples:
            label = one[1]
            if label == 1:
                pos += 1
            else:
                neg += 1
        print(f"#pos={pos}, #neg={neg}")

    def __len__(self):
        return len(self.samples)

    def _pad_his(self, his_ids, max_num):
        his_ids = his_ids[:max_num]
        his_mask = [1 for _ in range(len(his_ids))]
        his_mask += [0 for _ in range(max_num - len(his_ids))]
        his_ids += [torch.LongTensor([0])] * (max_num - len(his_ids))
        return his_ids, his_mask

    def _get_title_and_npy_single_old(self, itemid):
        exist_flag, npy_path = self._check_itemid_exists(itemid)
        if self.pickle_flag:
            with open(npy_path, 'rb') as f:
                cur_info = pickle.load(f)
        else:
            cur_info = load_obj_from_zip(npy_path)
        #cur_info = cur_info[str(itemid)]  # TODO
        title_fea = cur_info['title_fea']
        des_fea = cur_info['des_fea']
        des_mask = 1
        # des_mask = 1 if len(cur_info['des_score']['token']) > 2 else 0
        # if des_mask == 0:
        #     print("No des", itemid)
        return title_fea, des_fea, des_mask  # 768

    def _get_title_and_npy_old(self, itemid_list, max_num):
        fea_list, des_fea_list, des_mask_list = [], [], []
        for itemid in itemid_list:
            exist_flag, npy_path = self._check_itemid_exists(itemid)
            if exist_flag:
                if self.pickle_flag:
                    with open(npy_path, 'rb') as f:
                        cur_info = pickle.load(f)
                else:
                    cur_info = load_obj_from_zip(npy_path)
                #cur_info = cur_info[str(itemid)]  # TODO
                fea_list.append(cur_info['title_fea'])
                des_fea_list.append(cur_info['des_fea'])
                des_mask = 1
                # des_mask = 1 if len(cur_info['des_score']['token']) > 2 else 0
                # if des_mask == 0:
                #     print("No des in his", itemid)
                des_mask_list.append(des_mask)
        return fea_list[:max_num], des_fea_list[:max_num], des_mask_list[:max_num]  #  shape: #items, 768

    def _pad_his_fea(self, his_fea, his_des_fea, des_msk_list, max_num):
        his_fea = his_fea[:max_num]
        his_des_fea = his_des_fea[:max_num]
        des_msk_list = des_msk_list[:max_num]
        his_mask = [1 for _ in range(len(his_fea))]
        his_mask += [0 for _ in range(max_num - len(his_fea))]
        his_fea += [np.zeros(self.emb_hidden_size)] * (max_num - len(his_fea))
        his_des_fea += [np.zeros(self.emb_hidden_size)] * (max_num - len(his_des_fea))
        des_msk_list += [0 for _ in range(max_num - len(des_msk_list))]
        return his_fea, his_des_fea, his_mask, des_msk_list

    def __getitem__(self, item_idx):
        userid_itemid, label, current_itemid, his_browse, his_add_to_cart, \
            his_buy, sim_seq = self.samples[item_idx]
        current_title_fea, current_des_fea, cur_des_mask = self._get_title_and_npy_single_old(current_itemid)  # 768

        his_browse_fea, his_browse_des_fea, his_browse_des_msk = self._get_title_and_npy_old(his_browse, self.his_browse_num)
        his_add_fea, his_add_des_fea, his_add_des_msk = self._get_title_and_npy_old(his_add_to_cart, self.his_add_to_cart_num)
        his_buy_fea, his_buy_des_fea, his_buy_des_msk = self._get_title_and_npy_old(his_buy, self.his_buy_num)
        his_sim_fea, his_sim_des_fea, his_sim_des_msk = self._get_title_and_npy_old(sim_seq, self.his_sim_num)

        his_browse_fea, his_browse_des_fea, his_browse_mask, his_browse_des_msk = self._pad_his_fea(his_browse_fea, his_browse_des_fea, his_browse_des_msk, self.his_browse_num)
        his_add_fea, his_add_des_fea, his_add_mask, his_add_des_msk = self._pad_his_fea(his_add_fea, his_add_des_fea, his_add_des_msk, self.his_add_to_cart_num)
        his_buy_fea, his_buy_des_fea, his_buy_mask, his_buy_des_msk = self._pad_his_fea(his_buy_fea, his_buy_des_fea, his_buy_des_msk, self.his_buy_num)
        his_sim_fea, his_sim_des_fea, his_sim_mask, his_sim_des_msk = self._pad_his_fea(his_sim_fea, his_sim_des_fea, his_sim_des_msk, self.his_sim_num)
        #print("Query", query_raw_list, query_token_ids)
        return label, current_title_fea, current_des_fea, cur_des_mask, his_browse_fea, his_browse_des_fea, his_browse_mask, his_browse_des_msk, \
               his_add_fea, his_add_des_fea, his_add_mask, his_add_des_msk, his_buy_fea, his_buy_des_fea, his_buy_mask, his_buy_des_msk, \
               his_sim_fea, his_sim_des_fea, his_sim_mask, his_sim_des_msk

    @classmethod
    def pad(cls, batch):
        label_list, his_bro_msk_lst, his_add_msk_lst, his_buy_msk_lst, his_sim_msk_lst, his_msk_lst = [], [], [], [], [], []
        cur_fea_list, cur_des_fea_list, bro_fea_list, bro_des_fea_list, add_fea_list, add_des_fea_list, buy_fea_list, buy_des_fea_list = [], [], [], [], [], [], [], []
        sim_fea_list, sim_des_fea_list = [], []
        cur_des_msk_list, his_bro_des_msk_lst, his_add_des_msk_lst, his_buy_des_msk_lst, his_sim_des_msk_lst = [], [], [], [], []


        bs = len(batch)
        for (label, cur_fea, cur_des_fea, cur_des_msk, his_bro, his_bro_des, his_bro_msk, his_bro_des_msk,
             his_add, his_add_des, his_add_msk, his_add_des_msk,
             his_buy, his_buy_des, his_buy_msk, his_buy_des_msk,
             his_sim, his_sim_des, his_sim_msk, his_sim_des_msk) in batch:
            label_list.append(label)
            cur_fea_list.append(cur_fea)
            cur_des_fea_list.append(cur_des_fea)
            cur_des_msk_list.append(cur_des_msk)
            bro_fea_list.append(his_bro)
            bro_des_fea_list.append(his_bro_des)
            add_fea_list.append(his_add)
            add_des_fea_list.append(his_add_des)
            buy_fea_list.append(his_buy)
            buy_des_fea_list.append(his_buy_des)
            sim_fea_list.append(his_sim)
            sim_des_fea_list.append(his_sim_des)

            his_bro_msk_lst.append(his_bro_msk)
            his_add_msk_lst.append(his_add_msk)
            his_buy_msk_lst.append(his_buy_msk)
            his_sim_msk_lst.append(his_sim_msk)
            his_bro_des_msk_lst.append(his_bro_des_msk)
            his_add_des_msk_lst.append(his_add_des_msk)
            his_buy_des_msk_lst.append(his_buy_des_msk)
            his_sim_des_msk_lst.append(his_sim_des_msk)

            his_bro_flg = his_bro_msk[0]  # 是否拥有browse记录
            his_add_flg = his_add_msk[0]  # 是否拥有add_to_cart记录
            his_buy_flg = his_buy_msk[0]  # 是否拥有buy记录
            his_sim_flg = his_sim_msk[0]  # 是否拥有sim记录

            his_msk_lst.append([his_bro_flg, his_add_flg, his_buy_flg, his_sim_flg])


        label_tensor = torch.LongTensor(label_list)
        his_msk_tensor = torch.LongTensor(his_msk_lst)
        his_bro_msk_tensor = torch.LongTensor(his_bro_msk_lst)
        his_add_msk_tensor = torch.LongTensor(his_add_msk_lst)
        his_buy_msk_tensor = torch.LongTensor(his_buy_msk_lst)
        his_sim_msk_tensor = torch.LongTensor(his_sim_msk_lst)

        his_bro_des_msk_tensor = torch.LongTensor(his_bro_des_msk_lst)
        his_add_des_msk_tensor = torch.LongTensor(his_add_des_msk_lst)
        his_buy_des_msk_tensor = torch.LongTensor(his_buy_des_msk_lst)
        his_sim_des_msk_tensor = torch.LongTensor(his_sim_des_msk_lst)

        cur_fea_tensor = torch.FloatTensor(cur_fea_list)
        cur_des_fea_tensor = torch.FloatTensor(cur_des_fea_list)
        cur_des_msk_tensor = torch.LongTensor(cur_des_msk_list)
        bro_fea_tensor = torch.FloatTensor(bro_fea_list)
        bro_des_fea_tensor = torch.FloatTensor(bro_des_fea_list)
        add_fea_tensor = torch.FloatTensor(add_fea_list)
        add_des_fea_tensor = torch.FloatTensor(add_des_fea_list)
        buy_fea_tensor = torch.FloatTensor(buy_fea_list)
        buy_des_fea_tensor = torch.FloatTensor(buy_des_fea_list)
        sim_fea_tensor = torch.FloatTensor(sim_fea_list)
        sim_des_fea_tensor = torch.FloatTensor(sim_des_fea_list)

        return label_tensor, cur_fea_tensor, cur_des_fea_tensor, cur_des_msk_tensor, \
               his_msk_tensor, his_bro_msk_tensor, his_add_msk_tensor, his_buy_msk_tensor, his_sim_msk_tensor, \
               his_bro_des_msk_tensor, his_add_des_msk_tensor, his_buy_des_msk_tensor, his_sim_des_msk_tensor, \
               bro_fea_tensor, bro_des_fea_tensor, add_fea_tensor, add_des_fea_tensor, buy_fea_tensor, buy_des_fea_tensor, \
               sim_fea_tensor, sim_des_fea_tensor


    def _parse_token_old(self, token_in_title):
        tokens = ['[CLS]'] + self.tokenizer.tokenize(token_in_title) + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)[:self.max_token_num]
        return torch.LongTensor(token_ids)

    def _parse_token(self, token_in_title):
        token2ori_token = {}
        tok_in_title = token_in_title.split(' ')
        token2ori_token[0] = '[CLS]'
        tokens = ['[CLS]']
        for w in tok_in_title:
            sub_words = self.tokenizer.tokenize(w)
            if not sub_words:
                sub_words = ['[UNK]']
            for idx in range(len(sub_words)):
                token2ori_token[idx + len(tokens)] = w
            tokens += sub_words
        tokens = tokens[:self.max_token_num - 1] + ['[SEP]']
        token2ori_token[len(tokens) - 1] = '[SEP]'
        # tokens = ['[CLS]'] + self.tokenizer.tokenize(token_in_title) + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return torch.LongTensor(token_ids), token2ori_token

    def _load_title(self, title_path):
        itemid2title = {}  # itemid--> [title, des]
        all, has = 0, 0
        with open(title_path) as f:
            for line in f:
                tmp = eval(line.strip('\n'))
                # print("tmp", tmp)
                itemid = str(tmp[0])  # TODO  itemid from int to string
                title = tmp[1][0][1]
                des = tmp[1][1]
                if des is None: des = ""
                des = des.replace('\n\n', ' ').replace('\n', ' ')
                all += 1
                if title:
                    has += 1
                    itemid2title[itemid] = [title, des]
        print(f"#all_items={all}, #has_title={has}")
        for idx, (itemid, info) in enumerate(itemid2title.items()):
            if idx == 10: break
            print(itemid, info)
        return itemid2title

    def _filter_no_title(self, itemid_list, itemid2title, max_num):
        title_list, des_list = [], []
        for itemid in itemid_list:
            if str(itemid) in itemid2title:
                title, des = itemid2title[str(itemid)]
                title_list.append(title)
                des_list.append(des)
                if len(title_list) == max_num: break
        return title_list, des_list
        # return [itemid2title[str(itemid)] for itemid in itemid_list if str(itemid) in itemid2title][:max_num]

    def _check_itemid_exists_old(self, itemid):
        cur_npy_folder = os.path.join(self.npy_folder, str(itemid)[:5])
        if not os.path.exists(cur_npy_folder): return False, ""
        npy_path = os.path.join(cur_npy_folder, f"{itemid}.pkl")
        return os.path.exists(npy_path), npy_path

    def _check_itemid_exists(self, itemid):
        has_pkl_flag = (str(itemid) in self.has_pkl_item_set)
        cur_npy_folder = os.path.join(self.npy_folder, str(itemid)[:5])
        if self.pkl_name_substr_len > 0:
            npy_path = os.path.join(cur_npy_folder, f"{str(itemid)[:-self.pkl_name_substr_len]}.pkl")
        else:
            npy_path = os.path.join(cur_npy_folder, f"{itemid}.pkl")
        return has_pkl_flag, npy_path

    def _load_itemset(self):
        path = os.path.join(self.npy_folder, "itemid.pkl")
        with open(path, 'rb') as f:
            item_set = pickle.load(f)
        return item_set

    def _load_ctr_offline_new(self, ctr_path, debug):
        samples = []  # label, current_item, his_browse, his_add_to_cart, his_buy

        with open(ctr_path) as f:
            for lidx, line in enumerate(f):
                if lidx % 100000 == 0: print(lidx)
                tmp = json.loads(line.strip('\n'))
                #print('---', tmp)
                current_itemid = str(tmp[2])
                #exist_flag, _ = self._check_itemid_exists(current_itemid)
                #if not exist_flag: continue
                if current_itemid not in self.has_pkl_item_set: continue
                #print(current_itemid, len(itemid2title))
                his_browse_itemid_list = tmp[5]
                his_add_to_cart_itemid_list = tmp[6]
                his_buy_itemid_list = tmp[7]
                sim_seq_list = tmp[8]
                if sim_seq_list is None: sim_seq_list = []
                label = int(tmp[4])
                if label == 2: label = 1
                else: label = 0
                if len(his_browse_itemid_list) == 0 and len(his_add_to_cart_itemid_list) == 0 \
                        and len(his_buy_itemid_list) == 0 and len(sim_seq_list) == 0:
                    continue

                userid_itemid = str(tmp[0]) + "@" + current_itemid  # str(tmp[4])
                samples.append([userid_itemid, label, current_itemid, his_browse_itemid_list,
                               his_add_to_cart_itemid_list, his_buy_itemid_list, sim_seq_list])
                # print('---', label, itemid2title[current_itemid])
                # print('his bro', his_browse_itemid_list)
                # print('his add', his_add_to_cart_itemid_list)
                # print('his buy', his_buy_itemid_list)
                if debug and len(samples) >= 10000: break
        return samples

