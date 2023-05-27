#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import sklearn
from sklearn.datasets import load_svmlight_file
import xgboost as xgb

NUMBER_ROUNDS = 100
EARLY_STOPPING_ROUNDS = 8


X_START_IDX = 6 

# 0-date,1-stockno,
# 2-high,3-low,
# 4:features

def get_buffer_file(data_type="train",y_idx=2):
    return "data/reg_%s_%d.buffer"%(data_type,y_idx)


# 0.414,-0.382
def prepare_data(y_idx=2): 
    train_data = np.loadtxt('data/all_sample_train.txt', delimiter=',')
    train_X = np.ascontiguousarray(train_data[:, X_START_IDX:])
    train_Y = np.ascontiguousarray(train_data[:, y_idx])  # 2-最高价格，3-最低价格

    vali_data = np.loadtxt('data/all_validate.txt', delimiter=',')
    vali_X = np.ascontiguousarray(vali_data[:, X_START_IDX:])
    vali_Y = np.ascontiguousarray(vali_data[:, y_idx])

    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_vali = xgb.DMatrix(vali_X, label=vali_Y)

    # xg_train.save_binary(get_buffer_file("train",y_idx)) 
    # xg_vali.save_binary(get_buffer_file("vali",y_idx))
    return xg_train, xg_vali


def train(y_idx):
    tfile = get_buffer_file("train",y_idx)
    vfile = get_buffer_file("vali",y_idx)
    xg_train, xg_vali = (None, None)
    if os.path.exists(tfile):
        print("read from cache")
        xg_train = xgb.DMatrix(tfile)
        xg_vali = xgb.DMatrix(vfile)
    else:
        xg_train, xg_vali = prepare_data(y_idx)

    # setup parameters for xgboost
    # https://www.biaodianfu.com/xgboost.html
    param = {
        'objective': 'reg:squarederror',
        'eval_metric':'rmse',
        'eta': 0.1,
        'gamma': 0,
        'lambda': 1,
        'verbosity': 1,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
        'max_depth': 6
    }

    watchlist = [(xg_train, 'train'), (xg_vali, 'vali')]
    bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    bst.save_model("model_tmp/pointwise_%d_model.json"%(y_idx))

# python train_regression.py 2   # y-high
# python train_regression.py 3   # y-low
# python train_regression.py 4   # label-high
# python train_regression.py 5   # label-low
if __name__ == "__main__": 
    y_idx = int(sys.argv[1])
    assert y_idx in [2,3,4,5] , "y_idx not in [2,3,4,5]"
    # print(y_idx,sys.argv[1])
    train(y_idx)
