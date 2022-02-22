#coding:utf-8

import os
import sys
import json
import numpy as np
import collections
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
sys.path.append('./home/guozitao/project/cloud_card_BertSim/')
from threshold import ThrFor2021111009, ThrFor2021111209

np.set_printoptions(threshold=1e6)

label_ths_dict = ThrFor2021111209
#label_ths_dict = dict()

THR = 0.8
PRED_FALSE_LABEL = 'security'
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
stand_label_dict['all'] = 'all'
invalid_label_list = ['贷款,全款,首付,月供,利息,价格,按揭,免息,多少钱?', '配置,高配,油耗,天窗,自动,手动,混动,四驱,轮胎,底盘,行车记录仪。']

def calc_confusion_matrix(label_all, label_ths_dict, stand_label_dict):
    stand_label_dict.pop('all')
    label_id_dict = dict()
    real_label_num_dict = collections.defaultdict(int)
    max_idx = 0
    for idx, key in enumerate(stand_label_dict):
        label_id_dict[stand_label_dict[key]] = idx
        max_idx = idx
    label_id_dict['AABB'] = max_idx +1
    max_idx += 1
    cs_mat = np.zeros((max_idx+1, max_idx+1), dtype=np.int)
    for stand in label_all:
        if stand == 'all':
            continue
        for idx, pred in enumerate(label_all[stand][1]):       
            real_label = label_all[stand][3][idx]
            pred_label = stand_label_dict[stand]
            real_label_num_dict[real_label] += 1
            if my_round(pred, label_ths_dict[pred_label]) == 1:
                cs_mat[label_id_dict[real_label]][label_id_dict[pred_label]] += 1
    for idx, item in enumerate(stand_label_dict):
        print("{:<{}}{:<{}}{:<{}}".format(stand_label_dict[item] ,25, real_label_num_dict[stand_label_dict[item]]//18, 5, str(list(cs_mat[idx])), 100))
    print('{:<{}}{:<{}}{:<{}}'.format('AABB', 25, real_label_num_dict['AABB']//18, 5, str(list(cs_mat[max_idx])), 100))
    return

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
                print("%s\t%s\t%s\t%s\t%s"%(talk, label_all[stand][3][idx], label, label_all[stand][0][idx], pred))

def print_specific_label(stand, label_all, ths):
    
    for idx, pred in enumerate(label_all[stand][1]):
        pred_label = my_round(pred, ths)
        if pred_label != label_all[stand][0][idx]:
            print("%s\t%s\t%s\t%s"%(label_all[stand][4][idx], stand, label_all[stand][0][idx], pred))

with open('./data/standard_data.csv') as f:
    for line in f:
        if line.startswith('#'):
            continue
        label1, label2, stand = line.strip().split(',')
        stand = stand.lower()
        stand_label_dict[stand] = label2
        label_stand_dict[label2] = stand

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
    for idx, pred in enumerate(label_all[key][1]):
        if pred > temp_best_ths:
            label_all[key][2][idx] = 1
        else:
            label_all[key][2][idx] = 0
print(json.dumps(label_ths_dict))
#print_pred_false_label(label_all, label_ths_dict, stand_label_dict)
calc_confusion_matrix(label_all, label_ths_dict, stand_label_dict)
exit()

#for key in label_all:
#    for idx, pred in enumerate(label_all[key][1]):
#        ths = label_ths_dict[stand_label_dict[label_all[key][3][idx]]]
#        if pred > ths:
#            label_all[key][2][idx] = 1
#        else:
#            label_all[key][2][idx] = 0


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
        print("label_name:{:<{}}total_num:{:<{}}pos_num:{:<{}}F1:{:<{}}p:{:<{}}r:{:<{}}"
            .format(stand_label_dict[item[0]], 25, item[1][-1], 5, item[1][3], 5, item[1][0], 20, item[1][1], 20, item[1][2], 20))
