import random
random.seed(2022)
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

from vocab import Vocab

class NLPDatasetMask(Dataset):
    def __init__(self, item_path_str, tokenizer, max_token_num=512, mask_txt_ratio=0.1, max_mask_num=3, sample_ratio=1.1, debug=False):
        self.tokenizer = tokenizer
        self.vocab = self._get_vocab(self.tokenizer)
        self.max_token_num = max_token_num
        self.mask_txt_ratio = mask_txt_ratio
        self.max_mask_num = max_mask_num
        self.samples = []
        print('item_path', item_path_str.split(';'), len(item_path_str.split(';')))
        for item_path in item_path_str.split(';'):
            self.samples += self._load_title(item_path, sample_ratio, debug)
        for one_sample in self.samples[:10]:
            try:
                print("debug title", one_sample)
            except:
                continue

    def _get_vocab(self, tokenizer):
        vocab = Vocab()
        vocab.stoi = tokenizer.vocab
        try:
            vocab.itos = tokenizer.ids_to_tokens
        except:
            vocab.itos = tokenizer.inv_vocab
        vocab.words = [w for w in vocab.stoi]
        vocab.vocab_sz = len(vocab.itos)
        return vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item_idx):
        title = self.samples[item_idx]
        tokens = ['[CLS]'] + self.tokenizer.tokenize(title)[:self.max_token_num-2] + ['[SEP]']

        n_pred = min(len(tokens)-2, max(1, int(round((len(tokens)-2) * self.mask_txt_ratio))))
        n_pred = min(n_pred, self.max_mask_num)
        # print("n_pred", n_pred, len(tokens), list(range(1, len(tokens)-2)))
        masked_pos = random.sample(list(range(1, len(tokens)-1)), n_pred)  # 去掉[CLS]和[SEP]
        masked_pos.sort()

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:  # 对于sentence,将对应的masked_pos的字进行mask
            if random.random() < 0.8:  # 0.8, 80%的进行置换
                tokens[pos] = '[MASK]'
            elif random.random() < 0.5:  # 0.5, 10%随机换成另外一个字
                tokens[pos] = self.vocab.get_rand_word()
            else:  # 另外10%相当于保留原来的字，即可
                pass
        masked_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return torch.LongTensor(ids), torch.LongTensor(masked_pos), torch.LongTensor(masked_ids)

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


    def _load_title(self, item_path, sample_ratio, debug):
        samples = []
        itemset = set()
        with open(item_path) as f:
            for line in f:
                if random.random() >= sample_ratio: continue
                tmp = json.loads(line.strip('\n'))
                itemid = str(tmp[0])  # TODO  itemid from int to string
                if itemid in itemset: continue
                itemset.add(itemid)
                title = tmp[1]
                if title and len(title) > 0:
                    samples.append(title)
                    if debug and len(samples) >= 10000: break
        print("####samples", len(samples))
        return samples


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("cahya/bert-base-indonesian-522M")
    dataset = NLPDatasetMask(item_path="/Users/apple.yang/Documents/Data/ctr/shopee/title_info.txt", tokenizer=tokenizer)

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
