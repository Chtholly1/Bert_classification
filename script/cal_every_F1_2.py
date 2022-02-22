#coding:utf-8

import os
import re
import sys
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
sys.path.append('./home/guozitao/project/cloud_card_BertSim/')
from threshold import ThrFor20211221, ThrFor20211210, ThrFor20211119, ThrFor20211123, ThrFor20211124, ThrFor20211125, ThrFor20211130, ThrFor20211202, ThrFor20211216, ThrFor20211220
from label_black_word import ww_dict

label_ths_dict = ThrFor20211221
#label_ths_dict = dict()

THR = 0.8
PRED_FALSE_LABEL = 'space'
def my_round(prob, ths=0.9):
    if prob > ths:
        return 1
    else:
        return 0 
ths_list = list(np.arange(0.01, 1, 0.01))
label_all = dict()
label_all['all'] = [[], [], [], [], []]
stand_label_dict = dict()
label_stand_dict = dict()
talk_label_dict = dict()
stand_label_dict['all'] = 'all'
invalid_label_list = ['贷款,全款,首付,月供,利息,价格,按揭,免息,多少钱?', '配置,高配,油耗,天窗,自动,手动,混动,四驱,轮胎,底盘,行车记录仪,反光镜。']

def calc_confusion_matrix(label_all, label_ths_dict, stand_label_dict):
    label_id_dict = dict()
    max_idx = 0
    for idx, key in enumerate(stand_label_dict):
        label_id_dict[key] = idx
        max_idx = idx
    cs_mat = np.zeros((max_idx+1, max_idx+1))
    for stand in label_all:
        if stand == 'all':
            continue
        for idx, pred in enumerate(label_all[stand][1]):       
            real_label = label_all[stand][3][idx]
            pred_label = stand_label_dict[stand]
            if my_round(pred, label_ths_dict[pred_label]) == 1:
                cs_mat[label_id_dict[real_label]][label_id_dict[pred]] += 1
    print(cs_mat)

def print_pred_false_label(label_all, label_ths_dict, stand_label_dict):
    for stand in label_all:
        if stand == 'all':
            continue
        for idx, pred in enumerate(label_all[stand][1]):
            label = stand_label_dict[stand]
            pred_label = my_round(pred, label_ths_dict[label])
            if pred_label != label_all[stand][0][idx]:
                #print("%s\t%s\t%s\t%s"%(label_all[stand][4][idx], stand, label_all[stand][0][idx], pred))
                talk = label_all[stand][4][idx].replace('#','').replace(',', '，').replace(';','；').replace(':', '：').replace('?', '？').replace('(', '（').replace(')', '）').lower()
                print("%s\t%s\t%s\t%s\t%s"%(talk, talk_label_dict[talk], label, label_all[stand][0][idx], pred))

def print_specific_label(stand, label_all, ths):
    
    for idx, pred in enumerate(label_all[stand][1]):
        pred_label = my_round(pred, ths)
        if pred_label != label_all[stand][0][idx]:
            print("%s\t%s\t%s\t%s\t%s"%(label_all[stand][4][idx], stand, label_all[stand][3][idx], label_all[stand][0][idx], pred))

with open('./resource/data/standard_data.csv') as f:
    for line in f:
        if line.startswith('#'):
            continue
        label1, label2, stand = line.strip().split(',')
        stand = stand.lower()
        stand_label_dict[stand] = label2
        label_stand_dict[label2] = stand

with open('./resource/data/talk_to_stand.20211217') as f:
    for line in f:
        talk, stand, label = line.strip().split('\t')
        talk = talk.replace(' ','').replace('#','').replace(',', '，').replace(';','；').replace(':', '：').replace('?', '？').replace('(', '（').replace(')', '）').lower()
        talk_label_dict[talk] = label

#以标签为key,储存每个标签的预测值和实际标签
for line in sys.stdin:
    token = line.strip().split('\t')
    if token[0] == 'talk' or token[1] in invalid_label_list:
        continue
    if len(token) == 5:
        talk, stand, ori_label, label, pred = token
        #if ori_label != 'deep_price':
        #    continue
        pred = float(pred)
        stand = stand.replace('#','').replace(',', '，').replace(';','；').replace(':', '：').replace('?', '？').replace('(', '（').replace(')', '）').lower()
        if stand not in label_all:
            label_all[stand] = [[], [], [], [], []]
        label_all[stand][0].append(int(label))
        label_all[stand][1].append(pred)
        label_all[stand][2].append(my_round(pred, THR))
        label_all[stand][3].append(ori_label)
        label_all[stand][4].append(talk)
        label_all['all'][0].append(int(label))
        label_all['all'][1].append(pred)
        label_all['all'][2].append(my_round(pred, THR))
        label_all['all'][3].append(ori_label)
        label_all['all'][4].append(talk)
total_pos_sample = np.sum(label_all['all'][0])

#寻找每个标签的最佳阈值
for key in label_all:
    #temp_best_ths = 0
    #temp_best_F1 = 0
    #for ths in ths_list:
    #    for idx, pred in enumerate(label_all[key][1]):
    #        if pred > ths:
    #            label_all[key][2][idx] = 1
    #        else:
    #            label_all[key][2][idx] = 0
    #    if f1_score(label_all[key][0], label_all[key][2]) > temp_best_F1:
    #        temp_best_ths = ths
    #        temp_best_F1 = f1_score(label_all[key][0], label_all[key][2])
    #label_ths_dict[stand_label_dict[key]] = temp_best_ths
    temp_best_ths = label_ths_dict[stand_label_dict[key]]
    label = stand_label_dict[key]
    r_ww = None
    if label in ww_dict:
        r_ww = re.compile(ww_dict[label])
    for idx, pred in enumerate(label_all[key][1]):
        if r_ww and r_ww.search(label_all[key][4][idx]):
            label_all[key][2][idx] = 1 
        elif pred > temp_best_ths:
            label_all[key][2][idx] = 1
        else:
            label_all[key][2][idx] = 0
print(json.dumps(label_ths_dict))
#print_pred_false_label(label_all, label_ths_dict, stand_label_dict)
#print_specific_label(label_stand_dict[PRED_FALSE_LABEL], label_all, label_ths_dict[PRED_FALSE_LABEL])

real_all_res = [[],[]]
target_dict = dict()
for key in label_all:
    target_dict[key] = []
    target_dict[key].append(f1_score(label_all[key][0], label_all[key][2]))
    target_dict[key].append(precision_score(label_all[key][0], label_all[key][2]))
    target_dict[key].append(recall_score(label_all[key][0], label_all[key][2]))
    target_dict[key].append(np.sum(label_all[key][0]))
    target_dict[key].append(len(label_all[key][0]))
    if key != 'all':
        real_all_res[0].extend(label_all[key][0])
        real_all_res[1].extend(label_all[key][2])

print("label_name:{:<{}}total_num:{:<{}}pos_num:{:<{}}F1:{:<{}}p:{:<{}}r:{:<{}}".format('all', 25, len(real_all_res[0]), 5, total_pos_sample, 5, f1_score(real_all_res[0], real_all_res[1]), 20, precision_score(real_all_res[0], real_all_res[1]), 20, recall_score(real_all_res[0], real_all_res[1]), 20))

target_sort_list = sorted([(key, val) for key, val in target_dict.items()], key=lambda x:x[1][0], reverse=True)

for item in target_sort_list:
    if item[0] != 'all':
        #print("label_name:{:<{}}total_num:{:<{}}pos_num:{:<{}}F1:{:<{}}p:{:<{}}r:{:<{}}"
        #    .format(stand_label_dict[item[0]], 25, item[1][-1], 5, item[1][3], 5, item[1][0], 20, item[1][1], 20, item[1][2], 20))
        print("%s\t%s\t%s\t%s\t%s\t%s"%(stand_label_dict[item[0]], item[1][-1], item[1][3], item[1][0], item[1][1], item[1][2]))
