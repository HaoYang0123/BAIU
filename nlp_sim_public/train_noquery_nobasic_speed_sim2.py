import sys, os
import argparse
import time
import json
import gc
import numpy as np
import random
random.seed(2022)

import torch
import torch.distributed as dist
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

from dataset_noquery_nobasic_speed import NLPDataset
from dataset_noquery_nobasic_speed_npy_sim2 import NLPDataset as NLPDataset_npy

from model_add_base_query import NLPBertModel as NLPBertModel_v4
from model_noquery_nobasic_speed_sim2 import NLPBertModel
from utils import AverageMeter, meter_to_str

np.random.seed(44)
torch.manual_seed(44)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(44)

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
print("World size----------", WORLD_SIZE)
def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1

def is_distributed():
    return dist.is_available() and dist.is_initialized()

RBIT = 4

def split_train_file(ctr_path, split_num=1):
    if split_num == 1: return [ctr_path]
    ctr_path_list = []
    for idx in range(split_num):
        outpath = f"{ctr_path}_{idx}"
        ctr_path_list.append(outpath)
    fw_list = []
    for path in ctr_path_list:
        fw_list.append(open(path, 'w', encoding='utf8'))

    #with open(ctr_path, encoding='utf8') as f:
    #    lines = f.readlines()

    #random.shuffle(lines)
    #random.shuffle(lines)
    #random.shuffle(lines)

    with open(ctr_path, encoding='utf8') as f:
        for lidx, line in enumerate(f):
            fw_idx = lidx % split_num
            fw_list[fw_idx].write(line)

    for fw in fw_list:
        fw.close()

    # lines = []
    #del lines
    #gc.collect()
    return ctr_path_list

def main(param, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_num = torch.cuda.device_count()

    use_cuda = not param.cpu and torch.cuda.is_available()
    if use_cuda:
        print("use cuda...")
    if should_distribute():
        print('Using distributed PyTorch with {} backend'.format(param.backend))
        dist.init_process_group(backend=param.backend)

    if not os.path.exists(param.model_folder):
        os.mkdir(param.model_folder)


    if os.path.exists(param.test_title_path):
        if param.emb_flag == 'path':
            print("Use emb path")
            test_dataset = NLPDataset(title_path=param.test_title_path, emb_path=param.test_emb_path, ctr_path=param.test_ctr_path,
                                      tokenizer=tokenizer, his_browse_num=param.his_browse_num,
                                      his_add_to_cart_num=param.his_add_num, his_buy_num=param.his_buy_num,
                                      max_token_num=param.max_seq_len, emb_hidden_size=param.emb_hidden_size, debug=param.debug)
        else:
            print("Use npy folder")
            test_dataset = NLPDataset_npy(title_path=param.test_title_path, npy_folder=param.npy_folder, ctr_path=param.test_ctr_path,
                                      tokenizer=tokenizer, pickle_flag=param.pickle_flag, his_browse_num=param.his_browse_num,
                                      his_add_to_cart_num=param.his_add_num, his_buy_num=param.his_buy_num, his_sim_num=param.his_sim_num,
                                      max_token_num=param.max_seq_len, emb_hidden_size=param.emb_hidden_size, debug=param.debug)
        print("#testing samples", test_dataset.__len__())
    else:
        test_dataset = None
    val_dataset = test_dataset

    if test_dataset:
        test_dataloader = DataLoader(test_dataset, batch_size=param.batch_size*gpu_num, shuffle=False,
                                 num_workers=param.workers, collate_fn=NLPDataset_npy.pad)
    else:
        test_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(val_dataset, batch_size=param.batch_size*gpu_num, shuffle=False,
                                     num_workers=param.workers, collate_fn=NLPDataset_npy.pad)
    else:
        val_dataloader = None

    if param.bert_model.find('distilbert') >= 0:
        print("distibert model +++++", param.bert_model)
        bert_model = DistilBertModel.from_pretrained(param.bert_model)
    else:
        bert_model = BertModel.from_pretrained(param.bert_model)
    if param.model_flag == "new":
        model_flag = "bert"
        if param.bert_model.find('distilbert') >= 0:
            model_flag = "distilbert"
        model = NLPBertModel(bert_model=bert_model,
                             device=device, model_flag=model_flag, hidden_size=param.fc_hidden_size)
    else:
        raise NotImplementedError

    # ['v2', 'base']
    if param.checkpoint_flag == "new":
        if os.path.exists(param.checkpoint_path):
            print("load state", param.checkpoint_path)
            model.load_state_dict(torch.load(param.checkpoint_path)['model_state'])
            print("load complete")
    elif param.checkpoint_flag == 'v4':
        if os.path.exists(param.checkpoint_path):
            model_flag = "bert"
            if param.bert_model.find('distilbert') >= 0:
                model_flag = "distilbert"
            model_v4 = NLPBertModel_v4(bert_model=bert_model, device=device, model_flag=model_flag)
            model_v4.load_state_dict(torch.load(param.checkpoint_path)['model_state'])
            print("load complete with vvvv4444")
            model.bert = model_v4.bert
    else:
        raise NotImplementedError

    if is_distributed():
        print("use distributed")
        Distributor = nn.parallel.DistributedDataParallel if use_cuda \
            else nn.parallel.DistributedDataParallelCPU
        model = Distributor(model)
    elif gpu_num > 1:
        print("use multi gpus ========")
        model = torch.nn.DataParallel(model, device_ids=list(range(gpu_num)))

    print("model", model)

    param.eval_freq = param.eval_freq * param.split  # 0.1*10-->1
    split_eval_freq = 1
    if param.eval_freq >= 0.999999:
        split_eval_freq = param.eval_freq
        param.eval_freq = 100
    print("split_eval_freq+++++++", split_eval_freq)

    split_ctr_path_list = split_train_file(param.ctr_path, param.split)
    print("split_path_list", split_ctr_path_list)

    if param.fix_bert and param.model_flag == "new":
        if isinstance(model, torch.nn.DataParallel):
            for m in model.module.bert.parameters():
                m.requires_grad = False
        else:
            for m in model.bert.parameters():
                m.requires_grad = False

    param_optimizer = list(model.named_parameters())

    bert_param = ['bert', 'cls']  # 206 parameters
    # atten_param = ['cur_fc', 'his_bro_fc', 'his_add_fc', 'his_buy_fc', 'his_bro_fc_item', 'his_add_fc_item',
    #                 'his_buy_fc_item', 'his_fc', 'item_fc', 'user_fc', 'final_fc']  # 34 parameters
    bert_param_opt = [p for n, p in param_optimizer if any(nd in n for nd in bert_param)]
    atten_param_opt = [p for n, p in param_optimizer if not any(nd in n for nd in bert_param)]
    print("opt parameter size", len(bert_param_opt), len(atten_param_opt))

    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
    #                 ], 'weight_decay': param.weight_decay_finetune},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
    #                 ], 'weight_decay': 0.0}
    # ]

    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # new_param = ['cur_fc', 'his_bro_fc', 'his_add_fc', 'his_buy_fc', 'his_bro_fc_item', 'his_add_fc_item',
    #              'his_buy_fc_item', 'his_fc', 'item_fc', 'user_fc', 'final_fc']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
    #                 and not any(nd in n for nd in new_param)], 'weight_decay': param.weight_decay_finetune},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
    #                 and not any(nd in n for nd in new_param)], 'weight_decay': 0.0},
    #     {'params': [p for n, p in param_optimizer if n in new_param],
    #      'lr': param.lr_attention, 'weight_decay': param.weight_decay_finetune}
    # ]

    if param.fix_bert:
        optimizer_grouped_parameters = [{'params': atten_param_opt}]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=param.lr_attention)  # , warmup=param.warmup_proportion, t_total=total_train_steps
        optimizer.param_groups[0]['lr'] = param.lr_attention  # attention lr
        print("only optimize attention")
    else:
        optimizer_grouped_parameters = [{'params': bert_param_opt}, {'params': atten_param_opt}]
        # total_train_steps = int(train_dataset.__len__() / param.batch_size / param.gradient_accumulation_steps * param.epoches)
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                                     lr=param.learning_rate)  # , warmup=param.warmup_proportion, t_total=total_train_steps
        optimizer.param_groups[0]['lr'] = param.learning_rate  # bert lr
        optimizer.param_groups[1]['lr'] = param.lr_attention  # attention lr
        print("optimize bert+attention")

    for split_idx, one_ctr_path in enumerate(split_ctr_path_list):
        if param.emb_flag == 'path':
            print("Use emb path")
            train_dataset = NLPDataset(title_path=param.title_path, emb_path=param.emb_path, ctr_path=one_ctr_path,
                                       tokenizer=tokenizer,
                                       his_browse_num=param.his_browse_num,
                                       his_add_to_cart_num=param.his_add_num, his_buy_num=param.his_buy_num,
                                       max_token_num=param.max_seq_len, emb_hidden_size=param.emb_hidden_size, debug=param.debug)
        else:
            print("Use npy folder")
            train_dataset = NLPDataset_npy(title_path=param.title_path, npy_folder=param.npy_folder, ctr_path=one_ctr_path,
                                       tokenizer=tokenizer, pickle_flag=param.pickle_flag,
                                       his_browse_num=param.his_browse_num,
                                       his_add_to_cart_num=param.his_add_num, his_buy_num=param.his_buy_num, his_sim_num=param.his_sim_num,
                                       max_token_num=param.max_seq_len, emb_hidden_size=param.emb_hidden_size, debug=param.debug)
        print(f"#training samples in one_ctr_path={one_ctr_path} = {train_dataset.__len__()}")
        train_dataloader = DataLoader(train_dataset, batch_size=param.batch_size*gpu_num, shuffle=True,
                                  num_workers=param.workers, collate_fn=NLPDataset_npy.pad)
        print(f"start train on one_ctr_path={one_ctr_path}, all_splits={param.split}")
        train(model, optimizer, device, train_dataset, train_dataloader, test_dataset, test_dataloader, val_dataset, val_dataloader, param, split_eval_freq, split_idx=split_idx)
    logloss, auc = evaluate(model, test_dataset, test_dataloader, device, 100, 'Test_set', param)
    print(f"Evaluate on epoch=0, split_idx=Last, logloss={logloss}, auc={auc}")

def train(model, optimizer, device, train_dataset, train_dataloader, test_dataset, test_dataloader, val_dataset, val_dataloader, param, split_eval_freq,
          start_epoch=0, split_idx=0):
    torch.save({'epoch': 0, 'split': split_idx, 'model_state': model.state_dict(), 'logloss': 0,
                'auc': 0, 'max_seq_length': param.max_seq_len},
               os.path.join(param.model_folder, f'nlp_for_ctr_checkpoint_start.pt'))
    model.train()
    model = model.to(device)

    # loss_func = nn.CrossEntropyLoss()
    loss_func = F.binary_cross_entropy

    eval_step = 0
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
        eval_step = int(param.eval_freq * len(train_dataloader))
        eval_step = max(1, eval_step)
        print("eval_step", eval_step)
        eval_process_step = 0
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - start)
            batch = tuple(t.to(device) for t in batch)
            label_tensor = batch[0]

            ctr_prob, ctr_prob_sim = model(batch[1:])

            # binary loss
            loss = loss_func(ctr_prob, label_tensor.float())  # , reduction='sum'
            loss += loss_func(ctr_prob_sim, label_tensor.float())

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

                tr_loss += loss.item()
                losses.update(loss.item())
                batch_time.update(time.time() - start)
                start = time.time()
            # print("model bert", model.bert.encoder.layer[0].output.dense.weight.requires_grad,
            #       model.bert.encoder.layer[0].output.dense.weight)
            # print("model embedding", model.field_embeddings[0].weight)

            if (step + 1) % param.print_freq == 0:
                # if param.model_flag == 'new':
                #     print("model bert learning=", model.bert.encoder.layer[0].output.dense.weight.requires_grad)
                print("Epoch:{}-{}/{}, loss: [{}], [{}], [{}] ".\
                      format(epoch, step, len(train_dataloader), meter_to_str("Loss", losses, RBIT),
                             meter_to_str("Batch_Time", batch_time, RBIT),
                             meter_to_str("Data_Load_Time", data_time, RBIT)))
                #model.print_model_param()
            if (step + 1) % eval_step == 0 and val_dataset:
                print("Epoch:{}-{}/{}, loss: [{}], [{}], [{}] ". \
                      format(epoch, step, len(train_dataloader), meter_to_str("Loss", losses, RBIT),
                             meter_to_str("Batch_Time", batch_time, RBIT),
                             meter_to_str("Data_Load_Time", data_time, RBIT)))
                print("start evaluate")
                logloss, auc = evaluate(model, val_dataset, val_dataloader, device, epoch, 'Valid_set', param)
                print(f"Evaluate on epoch={epoch}, step={step+1}, logloss={logloss}, auc={auc}")
                if isinstance(model, torch.nn.DataParallel):
                    torch.save({'epoch': epoch, 'model_state': model.module.state_dict(), 'logloss': logloss,
                                'auc': auc, 'max_seq_length': param.max_seq_len},
                               os.path.join(param.model_folder, f'nlp_for_ctr_checkpoint_{epoch}_{split_idx}_{step+1}.pt'))
                else:
                    torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'logloss': logloss,
                                'auc': auc, 'max_seq_length': param.max_seq_len},
                               os.path.join(param.model_folder,
                                            f'nlp_for_ctr_checkpoint_{epoch}_{split_idx}_{step + 1}.pt'))
                eval_process_step += 1
                if not param.fix_bert and eval_process_step == param.num_step_bert:
                    print(f"---bert fixed on {eval_process_step}")
                    if isinstance(model, torch.nn.DataParallel):
                        for m in model.module.bert.parameters():
                            m.requires_grad = False
                    else:
                        for m in model.bert.parameters():
                            m.requires_grad = False
                    optimizer.param_groups[0]['lr'] = 0.0

        print('--------------------------------------------------------------')
        print("Epoch:{} completed, Total training's Loss: {}, Spend: {}minute".format(epoch, tr_loss,
                                                                                 (time.time() - train_start) / 60.0))
        if split_idx % split_eval_freq == 0:
            logloss, auc = evaluate(model, test_dataset, test_dataloader, device, epoch, 'Test_set', param)
            print(f"Evaluate on epoch={epoch}, split={split_idx}, logloss={logloss}, auc={auc}")
        else:
            logloss, auc = 0.0, 0.0
        if isinstance(model, torch.nn.DataParallel):
            torch.save({'epoch': epoch, 'split': split_idx, 'model_state': model.module.state_dict(), 'logloss': logloss,
                    'auc': auc, 'max_seq_length': param.max_seq_len},
                   os.path.join(param.model_folder, f'nlp_for_ctr_checkpoint_{epoch}_{split_idx}.pt'))
        else:
            torch.save({'epoch': epoch, 'split': split_idx, 'model_state': model.state_dict(), 'logloss': logloss,
                        'auc': auc, 'max_seq_length': param.max_seq_len},
                       os.path.join(param.model_folder, f'nlp_for_ctr_checkpoint_{epoch}_{split_idx}.pt'))

def evaluate(model, test_dataset, predict_dataloader, device, epoch_th, dataset_name, param):
    if not test_dataset: return 0, 0
    # print("***** Running prediction *****")
    model.eval()
    start = time.time()
    pred_ans, label_ans = [], []
    pred_sample_idx = 0
    fw = open(param.pred_outpath, 'w', encoding='utf8')
    with torch.no_grad():
        for step, batch in enumerate(predict_dataloader):
            batch = tuple(t.to(device) for t in batch)
            label_tensor = batch[0]

            y_pred_his, y_pred_sim = model(batch[1:])
            y_pred = (y_pred_his + y_pred_sim) / 2
            label_tmp = label_tensor.cpu().data.numpy()
            pred_tmp = y_pred.cpu().data.numpy()
            label_ans.append(label_tmp)
            pred_ans.append(pred_tmp)

            for one_pred, one_label in zip(pred_tmp, label_tmp):
                userid_itemid = test_dataset.samples[pred_sample_idx][0]
                fw.write(str(userid_itemid) + '\t' + str(float(one_pred)) + '\t' + str(int(one_label)) + '\n')
                pred_sample_idx += 1

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
    model.train()
    fw.close()
    return logloss, rocauc


if __name__ == '__main__':
    param = argparse.ArgumentParser(description='Train NLP for ctr Model')
    param.add_argument("--title-path", type=str,
                       default=r"", help="Data path for training")
    param.add_argument("--npy-folder", type=str,
                       default=r"", help="")
    param.add_argument("--emb-path", type=str, default=r"", help="")
    param.add_argument("--emb-flag", default='folder', choices=['folder', 'path'])
    param.add_argument("--emb-hidden-size", type=int, default=768)
    param.add_argument("--fc-hidden-size", type=int, default=128)
    param.add_argument("--test-emb-path", type=str, default=r"", help="")
    param.add_argument("--val-emb-path", type=str, default=r"", help="")
    param.add_argument("--pred-outpath", type=str, default=r"", help="")
    param.add_argument("--test-title-path", type=str,
                       default=r"", help="Data path for testing")
    param.add_argument("--ctr-path", type=str,
                       default=r"", help="")
    param.add_argument("--split", type=int, default=1, help="split for the ctr path")
    param.add_argument("--test-ctr-path", type=str,
                       default=r"", help="")
    param.add_argument("--test-split", type=int, default=1, help="split for the ctr path")
    param.add_argument("--val-ctr-path", type=str,
                       default=r"", help="")
    param.add_argument("--val-split", type=int, default=1, help="split for the ctr path")
    param.add_argument("--checkpoint-path", type=str,
                       default="", help="Pre-trained model path")
    param.add_argument("--checkpoint-flag", default="v2",
                       choices=['v1', 'v2', 'v4', 'new'], help="")
    param.add_argument("--model-flag", default="new",
                       choices=['base', 'new'], help="")
    param.add_argument("--batch-size", type=int,
                       default=2, help="Batch size of samples")
    param.add_argument("--workers", type=int,
                       default=0, help="Workers of dataLoader")
    param.add_argument("--epoches", type=int,
                       default=1, help="Epoches")
    param.add_argument("--num-step-bert", type=int,
                       default=1, help="Epoches")
    param.add_argument("--learning-rate", type=float,
                       default=5e-5, help="Learning rate for BERT when training")
    param.add_argument("--lr-attention", type=float,
                       default=5e-5, help="Learning rate for attention when training")
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
    param.add_argument("--eval-freq", type=float,
                       default=1.0, help="Frequency for printing training progress")

    # model parameters:
    param.add_argument("--his-browse-num", type=int, default=30)
    param.add_argument("--his-add-num", type=int, default=10)
    param.add_argument("--his-buy-num", type=int, default=10)
    param.add_argument("--his-sim-num", type=int, default=150)
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
    param.add_argument("--backend", type=str, default="")
    param.add_argument("--fix-bert", action='store_true')
    param.add_argument("--pickle-flag", action='store_true')

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
    main(param, tokenizer)
