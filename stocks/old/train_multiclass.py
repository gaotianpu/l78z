#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import sklearn
from sklearn.datasets import load_svmlight_file
import xgboost as xgb

NUMBER_ROUNDS = 80
EARLY_STOPPING_ROUNDS = 3

X_START_IDX = 4
Y_IDX = 3
TRAIN_BUFFER_FILE = "data/train_label_high.buffer"  
VALI_BUFFER_FILE = "data/vali_label_high.buffer"  

def prepare_data(): 
    train_data = np.loadtxt('data/all_train.txt', delimiter=',')
    train_X = np.ascontiguousarray(train_data[:, X_START_IDX:])
    train_Y = np.ascontiguousarray(train_data[:, Y_IDX])  # 2-最高价格，3-最低价格

    vali_data = np.loadtxt('data/all_validate.txt', delimiter=',')
    vali_X = np.ascontiguousarray(vali_data[:, X_START_IDX:])
    vali_Y = np.ascontiguousarray(vali_data[:, Y_IDX])

    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_vali = xgb.DMatrix(vali_X, label=vali_Y)

    xg_train.save_binary(TRAIN_BUFFER_FILE)
    xg_vali.save_binary(VALI_BUFFER_FILE)
    return xg_train, xg_vali


def train(): 
    xg_train, xg_vali = (None, None)
    if os.path.exists(TRAIN_BUFFER_FILE) and os.path.exists(TRAIN_BUFFER_FILE):
        xg_train = xgb.DMatrix(TRAIN_BUFFER_FILE)
        xg_vali = xgb.DMatrix(VALI_BUFFER_FILE)
    else:
        xg_train, xg_vali = prepare_data()

    # setup parameters for xgboost
    # https://www.biaodianfu.com/xgboost.html
    param = {
        'objective': 'multi:softmax',
        'eval_metric': 'auc',  #'mlogloss', 
        'num_class': 8,
        'eta': 0.5,
        'gamma': 0.1,
        'lambda': 2,
        'verbosity': 1,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
        'max_depth': 8
    }

    # watchlist = [(xg_train, 'train'), (xg_vali, 'vali')]
    # bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
    #                 early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    # bst.save_model("model/multiclass_high_model.json")

    param = {
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',  #'mlogloss', 
        'num_class': 8,
        'eta': 0.5,
        'gamma': 0.1,
        'lambda': 2,
        'verbosity': 1,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
        'max_depth': 10
    }

    watchlist = [(xg_train, 'train'), (xg_vali, 'vali')]
    bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    bst.save_model("model/multiclass_high_model_mlogloss.json")


if __name__ == "__main__":
    train()
