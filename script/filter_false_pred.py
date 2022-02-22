#coding:utf-8

import os
import sys
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

def my_round(prob, ths):
    if prob > ths:
        return 1
    else:
        return 0 
ths_list = list(range(0.9, 1, 0.1))

for line in sys.stdin:
    token = line.strip().split('\t')
    if token[0] == 'talk':
        continue
    if len(token) == 4:
        talk, stand, label, pred = token
        pred = float(pred)
        if label == '1':
            if pred <0.5:
                print("%s\t%.4f\t%s\t%s"%(label, pred, talk, stand))
        if label == '0':
            if pred > 0.5:
                print("%s\t%.4f\t%s\t%s"%(label, pred, talk, stand))
    elif len(token) == 5:
        talk, stand, ori_label, label, pred = token
        pred = float(pred)
        
        if label == '1':
            if pred <0.5:
                print("%s\t%.4f\t%s\t%s\t%s"%(label, pred, ori_label, talk, stand))
        if label == '0':
            if pred > 0.5:
                print("%s\t%.4f\t%s\t%s\t%s"%(label, pred, ori_label, talk, stand))   
