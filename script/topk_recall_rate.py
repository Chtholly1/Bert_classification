#coding:utf-8

import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
import collections

TOPK = 1

def load_stand_label(file_path):
    stand_label_dict = dict()
    with open(file_path) as f:
        for line in f:
            label1, label2, stand = line.strip().split(',', 3)
            stand_label_dict[stand.lower()] = label2
    return stand_label_dict

def my_round(prob, ths=0.9):
    if prob > ths:
        return 1
    else:
        return 0 
label_all = dict()

stand_label_path = './data/standard_data.csv'
stand_label_dict = load_stand_label(stand_label_path)

for line in sys.stdin:
    token = line.strip().split('\t')
    if token[0] == 'talk':
        continue
    if len(token) == 5:
        talk, stand, ori_label, label, pred = token
        stand = stand.replace('#','').replace(',', '，').replace(';','；').replace(':', '：').replace('?', '？').replace('(', '（').replace(')', '）').lower()
        if talk not in label_all:
            label_all[talk] = [[], [], [], '']
        label_all[talk][0].append(int(label))
        label_all[talk][1].append(float(pred))
        label_all[talk][2].append(stand_label_dict[stand])
        label_all[talk][3] = ori_label

recall_dict = dict()
recall_dict['all'] = [0,0, []]

for talk in label_all:
    pos_idx_list = []
    ori_label = label_all[talk][-1]
    if ori_label not in recall_dict:
        recall_dict[ori_label] = [0, 0, []]
    for idx, label in enumerate(label_all[talk][0]):
        if label == 1:
            pos_idx_list.append(idx)
    idx_list = range(len(label_all[talk][1]))
    pred_prob_idx_list = list(zip(idx_list, label_all[talk][1], label_all[talk][2]))
    pred_prob_sort_list = sorted(pred_prob_idx_list, key=lambda x:x[1], reverse=True)
    for i in range(TOPK):
        recall_dict[ori_label][2].append(pred_prob_sort_list[i][2])
        if pred_prob_sort_list[i][0] in pos_idx_list:
            recall_dict[ori_label][1] += 1
            recall_dict['all'][1] += 1
            break
    recall_dict[ori_label][0] += 1
    recall_dict['all'][0] += 1
    #if pred_prob_sort_list[0][0] not in pos_idx_list:
    #    print("%s\tlabel:%s\tpred:%s"%(talk, ori_label, pred_prob_sort_list[0][2]))
for item in recall_dict:
    print("label_name:{:<{}}total_num:{:<{}}r:{:<{}}"
            .format(item, 25, recall_dict[item][0], 5, recall_dict[item][1]/recall_dict[item][0], 20))

topk_count_dict = dict()
for key in recall_dict:
    topk_count_dict[key] = collections.defaultdict(int)
    for item in recall_dict[key][2]:
        topk_count_dict[key][item] += 1


for key in topk_count_dict:
    fre_sort_list = sorted([(key, val) for key, val in topk_count_dict[key].items()], key=lambda x:x[1], reverse=True)
    if not fre_sort_list:
        continue
    max_fre = fre_sort_list[0][1]
    sim_list = []
    for idx, item in enumerate(fre_sort_list):
        if item[1] > 0.2*max_fre:
            sim_list.append(item[0])
    sim_list = sim_list[:3]
    print("label:{:<{}}total_num:{:<{}}sim_label:{:<{}}".format(key, 25, recall_dict[key][0], 5, '\t'.join(sim_list), 100))
