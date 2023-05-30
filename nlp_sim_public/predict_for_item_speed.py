import sys, os
import argparse
import time
import json
import collections
import unicodedata
import six
from bert.bpe_helper import BPE
import sentencepiece as spm
from pythainlp import sent_tokenize
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from sklearn.metrics import log_loss, roc_auc_score

from dataset_item_speed import NLPDatasetForItem
from model_noquery_nobasic_speed_sim2 import NLPBertModel as NLPBertModel_ctr
from model_add_base_query import NLPBertModel
from utils import AverageMeter, meter_to_str, save_obj_into_zip, load_obj_from_zip
from config import DELETE_TOKEN_SET


np.random.seed(44)
torch.manual_seed(44)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(44)

RBIT = 4

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding='utf8') as reader:
        while True:
            token = reader.readline()
            if token.split(): token = token.split()[0]  # to support SentencePiece vocab file
            token = convert_to_unicode(token)
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    output = []
    for item in items:
        output.append(vocab[item])
    return output


class ThaiTokenizer(object):
    """Tokenizes Thai texts."""

    def __init__(self, vocab_file, spm_file):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.bpe = BPE(vocab_file)
        self.s = spm.SentencePieceProcessor()
        self.s.Load(spm_file)

    def tokenize(self, text):
        bpe_tokens = self.bpe.encode(text).split(' ')
        spm_tokens = self.s.EncodeAsPieces(text)

        tokens = bpe_tokens if len(bpe_tokens) < len(spm_tokens) else spm_tokens

        split_tokens = []

        for token in tokens:
            new_token = token

            if token.startswith('_') and not token in self.vocab:
                split_tokens.append('_')
                new_token = token[1:]

            if not new_token in self.vocab:
                split_tokens.append('<unk>')
            else:
                split_tokens.append(new_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

    def get_tokens(self):
        return list(self.vocab.keys())

def main(param, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_num = torch.cuda.device_count()

    unk_token = "[UNK]"
    if param.bert_model == 'monsoon-nlp/bert-base-thai':
        unk_token = "<unk>"
    pred_dataset = NLPDatasetForItem(title_path=param.title_path, tokenizer=tokenizer, infolder=param.infolder, outfolder=param.outfolder,
                                      unk_token=unk_token, max_token_num=param.max_seq_len, debug=param.debug)
    print("#predicting samples", pred_dataset.__len__())

    pred_dataloader = DataLoader(pred_dataset, batch_size=param.batch_size*gpu_num, shuffle=False,
                                  num_workers=param.workers, collate_fn=NLPDatasetForItem.pad)

    model_flag="distilbert"
    if param.bert_model.find('distilbert') >= 0:
        print("distibert model +++++", param.bert_model)
        bert_model = DistilBertModel.from_pretrained(param.bert_model)
    elif param.bert_model.find('roberta') >= 0:
        print("roberta model ------", param.bert_model)
        model_flag = "bert"
        bert_model = RobertaModel.from_pretrained(param.bert_model)
    else:
        model_flag = "bert"
        bert_model = BertModel.from_pretrained(param.bert_model)
    model = NLPBertModel(bert_model=bert_model,
                         device=device, model_flag=model_flag)

    model = model.to(device)
    if os.path.exists(param.checkpoint_path):
        if param.model_flag == "ori":
            print("load state", param.checkpoint_path)
            model.load_state_dict(torch.load(param.checkpoint_path)['model_state'])
            print("load complete")
        elif param.model_flag == "ctr":
            print("load state ctr model========", param.checkpoint_path)
            model_flag = "bert"
            if param.bert_model.find('distilbert') >= 0:
                model_flag = "distilbert"
            model_ctr = NLPBertModel_ctr(bert_model=bert_model,
                                 device=device, model_flag=model_flag, hidden_size=32)
            model_ctr.load_state_dict(torch.load(param.checkpoint_path)['model_state'])
            print("-----ccccc++++", model_ctr.bert.encoder.layer[0].output.dense.weight.requires_grad,
                  model_ctr.bert.encoder.layer[0].output.dense.weight)
            model.bert = model_ctr.bert
            print("load model_ctr complate and update bert parameters..........")
        else:
            raise NotImplementedError
    elif not param.debug:
        print("[Waring] ~!~~~~~ no input model")
    
        

    if not os.path.exists(param.outfolder):
        os.makedirs(param.outfolder)

    for itemid, title, des in pred_dataset.samples[:10]:
        tokens = ['[CLS]'] + tokenizer.tokenize(title) + ['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print("Debug for title", title)
        print("tokens", tokens)
        print("ids", ids)

        tokens_des = ['[CLS]'] + tokenizer.tokenize(des) + ['[SEP]']
        ids_des = tokenizer.convert_tokens_to_ids(tokens_des)
        print("Debug for des", des)
        print("tokens", tokens_des)
        print("ids", ids_des)

    multi_gpu = False
    if gpu_num > 1:
        print("multiple gpus=====")
        model = torch.nn.DataParallel(model, device_ids=list(range(gpu_num)))
        multi_gpu = True
    start = time.time()
    predict(model, pred_dataset, pred_dataloader, param, device, multi_gpu)
    print("time for predict", time.time()-start)


def predict(model, pred_dataset, pred_dataloader, param, device, multi_gpu):
    # print("***** Running prediction *****")
    model.eval()
    start_idx = 0

    def _get_token_score(score, idx2tokens):
        if np.sum(score[len(idx2tokens):]) != 0:
            print("[Warning] size not match...", score, len(idx2tokens))
        score = score[:len(idx2tokens)]  # 后面是Padding的token
        token_list, score_list = [], []
        for token_idx in range(len(idx2tokens)):
            one_score = score[token_idx]
            one_token = idx2tokens[token_idx]
            token_list.append(one_token)
            score_list.append(one_score)
        return {'token': token_list, 'score': ','.join([str(round(v, 6)) for v in score_list])}

    all_item_set = set()
    with torch.no_grad():
        for step, batch in enumerate(pred_dataloader):
            t_batch = tuple(t.to(device) for t in batch if type(t) == torch.Tensor)
            meta_info = [t for t in batch if type(t) != torch.Tensor]
            id, mask, des_id, des_mask = t_batch
            tit_dct_lst, des_dct_lst = meta_info

            atten_score, item_feat = model(id, mask)
            atten_des_score, item_des_feat = model(des_id, des_mask)

            atten_score = atten_score.cpu().data.numpy()
            item_feat = item_feat.cpu().data.numpy()
            atten_des_score = atten_des_score.cpu().data.numpy()
            item_des_feat = item_des_feat.cpu().data.numpy()
            for score, one_feat, des_score, des_one_feat, idx2tokens, des_idx2tokens in \
                    zip(atten_score, item_feat, atten_des_score, item_des_feat, tit_dct_lst, des_dct_lst):
                itemid = pred_dataset.samples[start_idx][0]

                # title_score_d = _get_token_score(score, idx2tokens)
                # des_score_d = _get_token_score(des_score, des_idx2tokens)

                write_d = {'title_fea': one_feat, 'des_fea': des_one_feat}
                           #'title_score': title_score_d, 'des_score': des_score_d}
                itm_folder = os.path.join(param.outfolder, str(itemid)[:5])
                if not os.path.exists(itm_folder):
                    os.mkdir(itm_folder)

                if not param.merge_pickle:
                    write_path = os.path.join(itm_folder, f"{itemid}.pkl")
                    with open(write_path, 'wb') as fw:
                        pickle.dump(write_d, fw)
                else:  # merge 10 items into one pkl file, why? because of limit file number
                    write_path = os.path.join(itm_folder, f"{str(itemid)[:-3]}.pkl")
                    if os.path.exists(write_path):
                        ori_d = load_obj_from_zip(write_path)
                    else:
                        ori_d = {}
                    new_data = {str(itemid): write_d}
                    ori_d.update(new_data)
                    save_obj_into_zip(ori_d, write_path)

                # title_path = os.path.join(param.outfolder, f"{itemid}.npy")
                # des_path = os.path.join(param.outfolder, f"{itemid}_des.npy")
                # np.save(title_path, one_feat)
                # np.save(des_path, des_one_feat)
                # with open(os.path.join(param.outfolder, f"{itemid}.json"), 'w', encoding='utf8') as fw:
                #     json.dump(title_score_d, fw)
                # with open(os.path.join(param.outfolder, f"{itemid}_des.json"), 'w', encoding='utf8') as fw:
                #     json.dump(des_score_d, fw)

                start_idx += 1
                all_item_set.add(str(itemid))
            if (step + 1) % param.print_freq == 0:
                print("process", step + 1, len(pred_dataloader))

    outpath = os.path.join(param.outfolder, "itemid.pkl")
    with open(outpath, 'wb') as fw:
        pickle.dump(all_item_set, fw)

if __name__ == '__main__':
    param = argparse.ArgumentParser(description='Train NLP for ctr Model')
    param.add_argument("--title-path", type=str,
                       default=r"", help="Data path for training")
    param.add_argument("--infolder", type=str, default=r"", help="")
    param.add_argument("--outfolder", type=str,
                       default=r"", help="Data path for training")
    param.add_argument("--checkpoint-path", type=str,
                       default="", help="Pre-trained model path")
    param.add_argument("--batch-size", type=int,
                       default=4, help="Batch size of samples")
    param.add_argument("--workers", type=int,
                       default=0, help="Workers of dataLoader")
    param.add_argument("--epoches", type=int,
                       default=1, help="Epoches")
    param.add_argument("--learning-rate", type=float,
                       default=5e-5, help="Learning rate when training")
    param.add_argument("--clip", type=float,
                       default=0.25, help="Learning rate for CRF")
    param.add_argument("--weight-decay-finetune", type=float,
                       default=1e-5, help="")
    param.add_argument("--warmup-proportion", type=float,
                       default=0.1, help="Proportion of training to perform linear learning rate warmup for")
    param.add_argument("--gradient-accumulation-steps", type=int,
                       default=1, help="Gradient accumulation steps")
    param.add_argument("--margin", type=float,
                       default=0.1, help="Margin for triplet loss")
    param.add_argument("--model-flag", type=str, default="ori", help="")
    param.add_argument("--model-folder", type=str,
                       default="./models", help="Folder for saved models")
    param.add_argument("--print-freq", type=int,
                       default=1, help="Frequency for printing training progress")
    param.add_argument("--eval-freq", type=float,
                       default=0.05, help="Frequency for printing training progress")

    # model parameters:
    param.add_argument("--topk", type=int, default=5)
    param.add_argument("--his-browse-num", type=int, default=30)
    param.add_argument("--his-add-num", type=int, default=10)
    param.add_argument("--his-buy-num", type=int, default=10)
    # param.add_argument("--loss-type", type=str, default="base",
    #                    help="Types of loss function (e.g., base, online semi-hard, online hardest, online hardest adv)")
    # param.add_argument("--img-input-size", type=int,
    #                    default=1792, help="Input size of image feature")
    # param.add_argument("--txt-output-dim", type=int,
    #                    default=768, help="Output dim for Bert")
    # param.add_argument("--final-dim", type=int,
    #                    default=128, help="Final feature dim")
    # param.add_argument("--dropout", type=float,
    #                    default=0.1, help="Dropout prob")

    param.add_argument("--debug", action='store_true')
    param.add_argument("--cpu", action='store_true')
    param.add_argument("--fix-bert", action='store_true')

    # bert parameters
    param.add_argument("--bert-model", type=str,
                       default="cahya/bert-base-indonesian-522M", help="Bert model name")
    param.add_argument("--bert-vocab-path", type=str,
                       default="", help="Bert vocab path")
    param.add_argument("--bert-model-path", type=str,
                       default="", help="Bert model path")
    param.add_argument("--max-seq-len", type=int,
                       default=512, help="Max number of word features for text")
    param.add_argument("--merge-pickle", action='store_true')
    # param.add_argument("--num-img-len", type=int,
    #                    default=80, help="Max number of region features for image")

    param = param.parse_args()
    print("Param", param)

    # tokenizer = BertTokenizer.from_pretrained(param.bert_model)

    if param.bert_vocab_path != "":  # get bert tokenizer
        tokenizer = BertTokenizer.from_pretrained(param.bert_vocab_path)
    else:
        if param.bert_model.find('distilbert') >= 0:
            print("load from distilbert", param.bert_model)
            tokenizer = DistilBertTokenizer.from_pretrained(param.bert_model)
        elif param.bert_model.find('roberta') >= 0:
            print("load from roberta", param.bert_model)
            tokenizer = RobertaTokenizer.from_pretrained(param.bert_model)
        else:
            tokenizer = BertTokenizer.from_pretrained(param.bert_model)
    if param.bert_model == 'monsoon-nlp/bert-base-thai':
        print("----switch into new tokenzier to support country=TH")
        tokenizer = ThaiTokenizer(vocab_file='th_model_vocab/th.wiki.bpe.op25000.vocab.txt',
                                  spm_file='th_model_vocab/th.wiki.bpe.op25000.model')
    main(param, tokenizer)
