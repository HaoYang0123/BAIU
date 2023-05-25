

from sklearn.metrics import log_loss, roc_auc_score
import numpy as np
import sys



inpath = "../../code/DeepCTR-Torch/examples/pred_item_user_query.txt"
if len(sys.argv)>1:
    inpath = sys.argv[1]
adpath = 'adid2num.txt'
t_num = 10000000  # 70% of ads
if len(sys.argv)>2:
    t_num = int(sys.argv[2])

adid2num = {}
retain_num, all_num = 0, 0
with open(adpath, encoding='utf8') as f:
    for line in f:
        tmp=line.strip('\n').split('\t')
        adid=tmp[0]
        num = int(tmp[1])
        adid2num[adid]=num
        all_num += 1
        if num <= t_num: retain_num += 1
print("###NUM", retain_num, all_num, retain_num/all_num)

labels, preds = [], []

with open(inpath, encoding='utf8') as f:
    for lidx,line in enumerate(f):
        if lidx%100000==0:print(lidx)
        tmp=line.strip('\n').split('\t')
        pred = float(tmp[0])
        l = int(tmp[1])
        adid = tmp[3]
        num=adid2num.get(adid,0)
        if num > t_num: continue
        preds.append(pred)
        labels.append(l)

print("###", len(labels), len(preds))
print("test LogLoss", round(log_loss(labels, preds), 4))
print("test AUC", round(roc_auc_score(labels, preds), 4))
