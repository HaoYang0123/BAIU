import random, json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

from vocab import Vocab

class NLPDatasetMask(Dataset):
    def __init__(self, item_path_str, tokenizer, tokenid2id_path, max_token_num=512, mask_txt_ratio=0.1, max_mask_num=3):
        self.tokenizer = tokenizer
        self.vocab = self._get_vocab(self.tokenizer)
        self.max_token_num = max_token_num
        self.mask_txt_ratio = mask_txt_ratio
        self.max_mask_num = max_mask_num
        self.samples = []
        print('item_path', item_path_str.split(';'), len(item_path_str.split(';')))
        with open(tokenid2id_path, encoding='utf8') as f:
            self.tokenid2new_id = json.load(f)
        for item_path in item_path_str.split(';'):
            tmp_list = self._load_title(item_path)
            self.samples += tmp_list
        print("#tokens", len(self.tokenid2new_id))  #, self.tokenid2new_id)
        #while True: pass

    def _get_vocab(self, tokenizer):
        vocab = Vocab()
        vocab.stoi = tokenizer.vocab
        vocab.itos = tokenizer.ids_to_tokens
        vocab.words = [w for w in vocab.stoi]
        vocab.vocab_sz = len(vocab.itos)
        return vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item_idx):
        itemid, id_in_title = self.samples[item_idx]
        id_in_title = [self.tokenid2new_id.get(v, 0) for v in id_in_title]
        tokens = self.tokenizer.convert_tokens_to_ids(['[CLS]']) + id_in_title + self.tokenizer.convert_tokens_to_ids(['[SEP]'])
        tokens = tokens[:self.max_token_num]

        n_pred = min(len(tokens)-2, max(1, int(round((len(tokens)-2) * self.mask_txt_ratio))))
        n_pred = min(n_pred, self.max_mask_num)
        # print("n_pred", n_pred, len(tokens), list(range(1, len(tokens)-2)))
        masked_pos = random.sample(list(range(1, len(tokens)-1)), n_pred)  # 去掉[CLS]和[SEP]
        masked_pos.sort()

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:  # 对于sentence,将对应的masked_pos的字进行mask
            if random.random() < 0.8:  # 0.8, 80%的进行置换
                tokens[pos] = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            elif random.random() < 0.5:  # 0.5, 10%随机换成另外一个字
                tokens[pos] = self.tokenizer.convert_tokens_to_ids([self.vocab.get_rand_word()])[0]
            else:  # 另外10%相当于保留原来的字，即可
                pass
        #masked_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
        #ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return torch.LongTensor(tokens), torch.LongTensor(masked_pos), torch.LongTensor(masked_tokens)

    @classmethod
    def pad(cls, batch):
        cur_max_num, mask_max_num = 0, 0
        bs = len(batch)
        for (ids, masked_pos, masked_ids) in batch:
            cur_max_num = max(cur_max_num, len(ids))
            mask_max_num = max(mask_max_num, len(masked_pos))

        cur_id_tensor = torch.zeros((bs, cur_max_num)).long()
        cur_mask_tensor = torch.zeros_like(cur_id_tensor).long()
        masked_pos_tensor = torch.zeros((bs, mask_max_num)).long()
        masked_mask_tensor = torch.zeros_like(masked_pos_tensor).long()
        masked_label_tensor = torch.zeros((bs, mask_max_num)).long()
        for idx, (ids, masked_pos, masked_ids) in enumerate(batch):
            cur_id_tensor[idx, :len(ids)] = ids
            cur_mask_tensor[idx, :len(ids)] = 1
            masked_pos_tensor[idx, :len(masked_pos)] = masked_pos
            masked_mask_tensor[idx, :len(masked_pos)] = 1
            masked_label_tensor[idx, :len(masked_ids)] = masked_ids
        return cur_id_tensor, cur_mask_tensor, masked_pos_tensor, masked_mask_tensor, masked_label_tensor


    def _load_title(self, item_path, min_token_len=3):
        samples = []

        with open(item_path) as f:
            for line in f:
                tmp = line.strip('\n').split('\t')
                itemid = tmp[0]
                title = tmp[1]
                id_in_title = title.split('|')
                if len(id_in_title) >= min_token_len:
                    samples.append([itemid, id_in_title])
        return samples


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = NLPDatasetMask(item_path_str="/Users/apple.yang/Documents/Data/ctr/kdd/titleid_tokensid_sample.txt",
                             tokenizer=tokenizer)

    print("#samples", dataset.__len__())
    # for idx in range(dataset.__len__()):
    #     print("input", dataset.samples[idx])
    #     ids, masked_pos, masked_ids = dataset.__getitem__(idx)
    #     print('ids', ids)
    #     print('masked_pos', masked_pos)
    #     print("masked_ids", masked_ids)

    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
                                  num_workers=0, collate_fn=NLPDatasetMask.pad)
    for idx, (cur_id_tensor, cur_mask_tensor, masked_pos_tensor, masked_mask_tensor, masked_label_tensor) in enumerate(train_dataloader):
        print(idx, '------')
        print('cur_id_tensor', cur_id_tensor)
        print('cur_mask_tensor', cur_mask_tensor)
        print('masked_pos_tensor', masked_pos_tensor)
        print('masked_mask_tensor', masked_mask_tensor)
        print('masked_label_tensor', masked_label_tensor)