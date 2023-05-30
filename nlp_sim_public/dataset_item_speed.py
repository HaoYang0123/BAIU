import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from functools import partial

class NLPDatasetForItem(Dataset):
    def __init__(self, title_path, tokenizer, infolder, outfolder, unk_token='[UNK]', max_token_num=512, debug=False):
        self.tokenizer = tokenizer
        self.max_token_num = max_token_num
        self.unk_token = unk_token
        self.samples = self._load_title(title_path, infolder, outfolder, debug)
        if debug:
            self.samples = self.samples[:10000]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item_idx):
        itemid, title, des = self.samples[item_idx]
        token_ids, tit_token_2ori_token = self._parse_token(title)  # one list = [id1, id2, ..., idn]
        token_des_ids, des_token_2ori_token = self._parse_token(des)  # one list = [id1, id2, ..., idm]
        return token_ids, token_des_ids, tit_token_2ori_token, des_token_2ori_token

    def _parse_token(self, token_in_title):
        token2ori_token = {}
        tok_in_title = token_in_title.split(' ')
        token2ori_token[0] = '[CLS]'
        tokens = ['[CLS]']
        for w in tok_in_title:
            sub_words = self.tokenizer.tokenize(w)
            if not sub_words:
                sub_words = [self.unk_token]
            for idx in range(len(sub_words)):
                token2ori_token[idx + len(tokens)] = w
            tokens += sub_words
        tokens = tokens[:self.max_token_num - 1] + ['[SEP]']
        token2ori_token[len(tokens) - 1] = '[SEP]'
        # tokens = ['[CLS]'] + self.tokenizer.tokenize(token_in_title) + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return torch.LongTensor(token_ids), token2ori_token

    def get_info(self, item_idx):
        itemid, title = self.samples[item_idx]
        token2ori_token = {}
        tok_in_title = title.split(' ')
        token2ori_token[0] = '[CLS]'
        tokens = ['[CLS]']
        for w in tok_in_title:
            sub_words = self.tokenizer.tokenize(w)
            if not sub_words:
                sub_words = [self.unk_token]
            for idx in range(len(sub_words)):
                token2ori_token[idx+len(tokens)] = w
            tokens += sub_words
        tokens = tokens[:self.max_token_num-1] + ['[SEP]']
        token2ori_token[len(tokens)-1] = '[SEP]'
        return itemid, token2ori_token

    @classmethod
    def pad(cls, batch):
        cur_max_num, cur_des_max_num = 0, 0
        bs = len(batch)
        tit_dct_lst, des_dct_lst = [], []
        for cur_ids, cur_des_ids, tit_token_2ori_token, des_token_2ori_token in batch:
            cur_max_num = max(cur_max_num, len(cur_ids))
            cur_des_max_num = max(cur_des_max_num, len(cur_des_ids))
            tit_dct_lst.append(tit_token_2ori_token)
            des_dct_lst.append(des_token_2ori_token)

        cur_id_tensor = torch.zeros((bs, cur_max_num), dtype=torch.long)
        cur_mask_tensor = torch.zeros_like(cur_id_tensor, dtype=torch.long)
        cur_des_id_tensor = torch.zeros((bs, cur_des_max_num), dtype=torch.long)
        cur_des_mask_tensor = torch.zeros_like(cur_des_id_tensor, dtype=torch.long)

        for idx, (cur_ids, cur_des_ids, _, _) in enumerate(batch):
            cur_id_tensor[idx, :len(cur_ids)] = cur_ids
            cur_mask_tensor[idx, :len(cur_ids)] = 1
            cur_des_id_tensor[idx, :len(cur_des_ids)] = cur_des_ids
            cur_des_mask_tensor[idx, :len(cur_des_ids)] = 1

        return cur_id_tensor, cur_mask_tensor, cur_des_id_tensor, cur_des_mask_tensor, tit_dct_lst, des_dct_lst


    # def _parse_token(self, token_in_title):
    #     tok_in_title = token_in_title.split(' ')
    #     tokens = ['[CLS]']
    #     for w in tok_in_title:
    #         sub_words = self.tokenizer.tokenize(w)
    #         if not sub_words:
    #             sub_words = ['[UNK]']
    #         tokens += sub_words
    #     tokens = tokens[:self.max_token_num - 1] + ['[SEP]']
    #     #tokens = ['[CLS]'] + self.tokenizer.tokenize(token_in_title) + ['[SEP]']
    #     token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    #     return torch.LongTensor(token_ids)

    def _check_npy_exist(self, itemid, infolder_list):
        for infolder in infolder_list:
            path = os.path.join(infolder, f"{itemid}.npy")
            if os.path.exists(path):
                return True
        return False

    def _load_title(self, title_path_str, infolder, outfolder, debug):
        infolder_list = infolder.split(';')
        itemid2title = {}
        all, has = 0, 0
        for title_path in title_path_str.split(";"):
            print("title", title_path)
            if not os.path.exists(title_path):
                print(f"[Warning] {title_path} not exists...")
                continue
            with open(title_path) as f:
                for lidx, line in enumerate(f):
                    tmp = json.loads(line.strip('\n'))
                    if tmp is None: continue
                    itemid = str(tmp[0])  # TODO  itemid from int to string
                    title = tmp[1]
                    if title is None: continue
                    des = tmp[-1]
                    if des is None: des = ""
                    des = des.replace('[', '').replace('{"t":"', '').replace('\n"}', '')
                    des = des.replace('\\n\\n', ' ').replace('\\r\\n', ' ').replace('\\n', ' ')
                    all += 1
                    if title:
                        has += 1
                        itemid2title[itemid] = [title, des]
                        if lidx <= 10:
                            try:
                                print('-->', title_path, lidx, itemid, [title, des])
                            except:
                                pass
                        if debug and lidx >= 20000: break

        print(f"#all_items={all}, #has_title={has}, #delete-same-items={len(itemid2title)}")
        samples = []
        for itemid in itemid2title:
            title, des = itemid2title[itemid]
            samples.append([itemid, title, des])
        for idx, (itemid, title, des) in enumerate(samples):
            if idx>=10: break
            try:
                print('title', itemid, [title, des])
            except: continue
        print("#need process samples", len(samples))
        return samples


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("cahya/bert-base-indonesian-522M")
    dataset = NLPDatasetForItem(title_path="/Users/apple.yang/Documents/Data/ctr/shopee/title_info.txt",
                                tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
                                  num_workers=0, collate_fn=NLPDatasetForItem.pad)
    for step, (id_tensor, mask_tensor) in enumerate(train_dataloader):
        print(step, id_tensor, mask_tensor)
