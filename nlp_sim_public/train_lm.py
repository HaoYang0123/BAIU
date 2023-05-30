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

import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer, BertModel
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from sklearn.metrics import log_loss, roc_auc_score

from dataset_mask import NLPDatasetMask
from model_add_base_query import NLPBertModel
from utils import AverageMeter, meter_to_str


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

    if not os.path.exists(param.model_folder):
        os.mkdir(param.model_folder)

    train_dataset = NLPDatasetMask(item_path_str=param.title_path, tokenizer=tokenizer,
                                   max_token_num=param.max_seq_len, mask_txt_ratio=param.mask_txt_ratio,
                                   max_mask_num=param.max_mask_num, sample_ratio=param.sample_ratio, debug=param.debug)
    print("#training samples", train_dataset.__len__())

    train_dataloader = DataLoader(train_dataset, batch_size=param.batch_size, shuffle=True,
                                  num_workers=param.workers, collate_fn=NLPDatasetMask.pad)

    model_flag = "bert"
    if param.bert_model.find('distilbert') >= 0:
        model_flag = "distilbert"
        print("distibert model +++++", param.bert_model)
        bert_model = DistilBertModel.from_pretrained(param.bert_model)
    else:
        bert_model = BertModel.from_pretrained(param.bert_model)
    model = NLPBertModel(bert_model=bert_model, model_flag=model_flag,
                         device=device)
    if os.path.exists(param.checkpoint_path):
        print("load state", param.checkpoint_path)
        model.load_state_dict(torch.load(param.checkpoint_path)['model_state'])
        print("load complete")

    for title in train_dataset.samples[:10]:
        tokens = ['[CLS]'] + tokenizer.tokenize(title) + ['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print("Debug for title", title)
        print("tokens", tokens)
        print("ids", ids)

    train(model, device, train_dataset, train_dataloader, param)

def train(model, device, train_dataset, train_dataloader, param,
          start_epoch=0):
    model.train()
    model = model.to(device)

    loss_func = nn.CrossEntropyLoss(reduction='none')

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
                    ], 'weight_decay': param.weight_decay_finetune},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
                    ], 'weight_decay': 0.0}
    ]
    # total_train_steps = int(train_dataset.__len__() / param.batch_size / param.gradient_accumulation_steps * param.epoches)
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=param.learning_rate)  # , warmup=param.warmup_proportion, t_total=total_train_steps
    global_step_th = int(train_dataset.__len__() / param.batch_size / param.gradient_accumulation_steps * start_epoch)

    for epoch in range(start_epoch, param.epoches):
        model.train()
        optimizer.zero_grad()

        train_start = time.time()
        tr_loss = 0.0
        start = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - start)
            batch = tuple(t.to(device) for t in batch)
            cur_id_tensor, cur_mask_tensor, masked_pos_tensor, masked_mask_tensor, masked_label_tensor = batch

            token_pred_prob = model.forward_lm(cur_id_tensor, cur_mask_tensor, masked_pos_tensor)

            # binary loss
            loss_each = loss_func(token_pred_prob.transpose(1, 2).float(), masked_label_tensor)
            #print('loss_each', loss_each)
            loss_each = loss_each * masked_mask_tensor
            #print('loss_each 222', loss_each)
            loss = torch.sum(loss_each, dim=-1) / (torch.sum(masked_mask_tensor, dim=-1) + 1e-6)
            #print('mask', masked_mask_tensor)
            #print('loss', loss)
            loss = torch.mean(loss)
            #print("loss 222", loss)

            # cross-entropy loss
            # ctr_prob_exp = torch.stack([1 - ctr_prob, ctr_prob], dim=-1)
            # loss = loss_func(ctr_prob_exp, label_tensor)

            if param.gradient_accumulation_steps > 1:
                loss = loss / param.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % param.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                # lr_this_step = param.learning_rate * warmup_linear(global_step_th / total_train_steps, param.warmup_proportion)
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr_this_step
                #torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), param.clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step_th += 1

                tr_loss += loss.item()
                losses.update(loss.item())
                batch_time.update(time.time() - start)
                start = time.time()
            # print("model bert", model.bert.encoder.layer[0].output.dense.weight.requires_grad,
            #       model.bert.encoder.layer[0].output.dense.weight)

            if (step + 1) % param.print_freq == 0:
                #print("model bert learning=", model.bert.encoder.layer[0].output.dense.weight.requires_grad)
                print("Epoch:{}-{}/{}, loss: [{}], [{}], [{}] ".\
                      format(epoch, step, len(train_dataloader), meter_to_str("Loss", losses, RBIT),
                             meter_to_str("Batch_Time", batch_time, RBIT),
                             meter_to_str("Data_Load_Time", data_time, RBIT)))
                #model.print_model_param()

        print('--------------------------------------------------------------')
        print("Epoch:{} completed, Total training's Loss: {}, Spend: {}minute".format(epoch, tr_loss,
                                                                                 (time.time() - train_start) / 60.0))
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'logloss': tr_loss,
                    'max_seq_length': param.max_seq_len},
                   os.path.join(param.model_folder, f'nlp_lm_checkpoint_{epoch}.pt'))

def evaluate(model, test_dataset, predict_dataloader, device, epoch_th, dataset_name, param):
    if not test_dataset: return 0, 0
    # print("***** Running prediction *****")
    model.eval()
    start = time.time()
    pred_ans, label_ans = [], []
    with torch.no_grad():
        for step, batch in enumerate(predict_dataloader):
            batch = tuple(t.to(device) for t in batch)
            label_tensor, cur_id_tensor, cur_mask_tensor, his_msk_tensor, his_bro_msk_tensor, his_add_msk_tensor, his_buy_msk_tensor, \
            his_bro_id_tensor, his_add_id_tensor, his_buy_id_tensor, his_bro_m_tensor, his_add_m_tensor, his_buy_m_tensor = batch

            y_pred = model(cur_id_tensor, cur_mask_tensor, his_msk_tensor, his_bro_msk_tensor, his_add_msk_tensor,
                             his_buy_msk_tensor, his_bro_id_tensor, his_add_id_tensor, his_buy_id_tensor, his_bro_m_tensor,
                             his_add_m_tensor, his_buy_m_tensor)
            label_ans.append(label_tensor.cpu().data.numpy())
            pred_ans.append(y_pred.cpu().data.numpy())
            if (step + 1) % param.print_freq == 0:
                pred_ans_2 = np.concatenate(pred_ans).astype("float64")
                label_ans_2 = np.concatenate(label_ans).astype("int32")
                logloss = round(log_loss(label_ans_2, pred_ans_2), RBIT)
                rocauc = round(roc_auc_score(label_ans_2, pred_ans_2), RBIT)
                print(f"evaluate on epoch={epoch_th}, inter-step={step}/{len(predict_dataloader)}, logloss={logloss}, auc={rocauc}")
    pred_ans = np.concatenate(pred_ans).astype("float64")
    label_ans = np.concatenate(label_ans).astype("int32")
    # print("pred_ans", pred_ans)
    # print("label_ans", label_ans)

    logloss = round(log_loss(label_ans, pred_ans), RBIT)
    rocauc = round(roc_auc_score(label_ans, pred_ans), RBIT)
    return logloss, rocauc


if __name__ == '__main__':
    param = argparse.ArgumentParser(description='Train NLP for ctr Model')
    param.add_argument("--title-path", type=str,
                       default=r"", help="Data path for training")
    param.add_argument("--sample-ratio", type=float, default=1.1, help="sample ratio for training")
    param.add_argument("--checkpoint-path", type=str,
                       default="", help="Pre-trained model path")
    param.add_argument("--batch-size", type=int,
                       default=4, help="Batch size of samples")
    param.add_argument("--workers", type=int,
                       default=0, help="Workers of dataLoader")
    param.add_argument("--epoches", type=int,
                       default=1, help="Epoches")
    param.add_argument("--learning-rate", type=float,
                       default=5e-5, help="Learning rate for BERT when training")
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
    param.add_argument("--model-folder", type=str,
                       default="./models", help="Folder for saved models")
    param.add_argument("--print-freq", type=int,
                       default=1, help="Frequency for printing training progress")

    # model parameters:
    param.add_argument("--mask-txt-ratio", type=float, default=0.1)
    param.add_argument("--max-mask-num", type=int, default=3)
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

    # bert parameters
    param.add_argument("--bert-model", type=str,
                       default="cahya/bert-base-indonesian-522M", help="Bert model name")  #cahya/distilbert-base-indonesian
    param.add_argument("--bert-vocab-path", type=str,
                       default="", help="Bert vocab path")
    param.add_argument("--bert-model-path", type=str,
                       default="", help="Bert model path")
    param.add_argument("--max-seq-len", type=int,
                       default=512, help="Max number of word features for text")
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
        else:
            tokenizer = BertTokenizer.from_pretrained(param.bert_model)
    if param.bert_model == 'monsoon-nlp/bert-base-thai':
        print("----switch into new tokenzier to support country=TH")
        tokenizer = ThaiTokenizer(vocab_file='th_model_vocab/th.wiki.bpe.op25000.vocab.txt',
                                  spm_file='th_model_vocab/th.wiki.bpe.op25000.model')
    main(param, tokenizer)
