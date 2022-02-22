#coding:utf-8

import os
import sys
import json
import random
import numpy as np
import scipy.optimize
from functools import partial
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

def acc(x, y):
    x = np.array(x)
    y = np.array(y)
    score = np.sum(x ==  y)
    return score/x.shape[0]

def CE(X, Y):
    cross_loss = 0
    for idx, x in enumerate(X):
        cross_loss += Y[idx]*np.log(X[idx]+1e-6) + (1-Y[idx])*np.log(1-X[idx]+1e-6)
    return cross_loss

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            else:
                X_p[i] = 1
        ll = acc(X_p, y)
        return -ll

    def mult_class_loss(self, coef, X, y):
        X_p = np.copy(X)
        coef_np = np.array(coef)#.reshape(-1,1) 
        X_p = X_p * coef_np
        pred = np.argmax(X_p, axis=1)        
        score = acc(pred, y)
        return -score

    def fit(self, X, y):
        initial_coef = [0.9]*X.shape[1]    
        #initial_coef = [ random.random() for x in range(X.shape[1])]
        print(initial_coef)
        #self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        self.coef_ = scipy.optimize.minimize(self.mult_class_loss, initial_coef, (X,y), method='nelder-mead')
        print(self.coef_)

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            else:
                X_p[i] = 1
        return X_p

if __name__ == '__main__':
    text_list, X, Y = [], [], []
    c = OptimizedRounder()
    for line in sys.stdin:
        text, y, x = line.strip().split('\t')
        Y.append(int(y))
        X.append(eval(x))
    Y = np.array(Y)
    X = np.array(X)
    c.fit(X, Y)
