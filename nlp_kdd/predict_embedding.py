import sys, os
import argparse
import time
import json
import pickle
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
from model_lm import NLPBertModel
from utils import AverageMeter, meter_to_str


np.random.seed(44)
torch.manual_seed(44)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(44)

RBIT = 4

def main(param, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(param.checkpoint_path):
        print("no input model...")
        sys.exit(-1)

    if not os.path.exists(param.out_folder):
        os.mkdir(param.out_folder)

    pred_dataset = NLPDatasetMask(item_path_str=param.title_path, tokenizer=tokenizer, tokenid2id_path=param.tokenid2id_path,
                                   max_token_num=param.max_seq_len, mask_txt_ratio=param.mask_txt_ratio,
                                   max_mask_num=param.max_mask_num)
    print("#predicting samples", pred_dataset.__len__())

    pred_dataloader = DataLoader(pred_dataset, batch_size=param.batch_size, shuffle=False,
                                  num_workers=param.workers, collate_fn=NLPDatasetMask.pad)

    model_flag = "bert"
    if param.bert_model.find('distilbert') >= 0:
        model_flag = "distilbert"
        print("distibert model +++++", param.bert_model)
        bert_model = DistilBertModel.from_pretrained(param.bert_model)
    else:
        bert_model = BertModel.from_pretrained(param.bert_model)
    print("config", bert_model.config)
    model = NLPBertModel(bert_model=bert_model, device=device, model_flag=model_flag)
    model.load_state_dict(torch.load(param.checkpoint_path)['model_state'])
    model = model.to(device)
    print("load model complete.")
    evaluate(model, device, pred_dataset, pred_dataloader, param)

def evaluate(model, device, test_dataset, predict_dataloader, param):
    if not test_dataset: return
    # print("***** Running prediction *****")
    model.eval()
    start = time.time()
    sample_idx = 0
    with torch.no_grad():
        for step, batch in enumerate(predict_dataloader):
            batch = tuple(t.to(device) for t in batch)
            cur_id_tensor, cur_mask_tensor, masked_pos_tensor, masked_mask_tensor, masked_label_tensor = batch
            atten_prob, cur_feat = model.forward_emb(cur_id_tensor, cur_mask_tensor)
            atten_prob = atten_prob.cpu().data.numpy()
            cur_feat = cur_feat.cpu().data.numpy()
            for (one_feat, one_prob) in zip(cur_feat, atten_prob):
                itemid, tokens = test_dataset.samples[sample_idx]
                sample_idx += 1
                write_d = {'fea': one_feat, 'score': one_prob, 'tokens': tokens}
                write_folder = os.path.join(param.out_folder, str(itemid)[:5])
                if not os.path.exists(write_folder): os.mkdir(write_folder)
                write_path = os.path.join(write_folder, f"{itemid}.pkl")
                with open(write_path, 'wb') as fw:
                    pickle.dump(write_d, fw)

            if (step + 1) % param.print_freq == 0:
                print(f"evaluate on {step+1}")


if __name__ == '__main__':
    param = argparse.ArgumentParser(description='Train NLP for ctr Model')
    param.add_argument("--title-path", type=str,
                       default=r"", help="Data path for training")
    param.add_argument("--tokenid2id-path", type=str, default=r"", help="")
    param.add_argument("--checkpoint-path", type=str,
                       default="", help="Pre-trained model path")
    param.add_argument("--out-folder", type=str,
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

    param.add_argument("--debug", action='store_true')
    param.add_argument("--cpu", action='store_true')

    # bert parameters
    param.add_argument("--bert-model", type=str,
                       default="bert-base-uncased", help="Bert model name")  #cahya/distilbert-base-indonesian
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
    main(param, tokenizer)