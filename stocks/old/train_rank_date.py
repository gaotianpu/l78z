#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import sklearn
from sklearn.datasets import load_svmlight_file
import xgboost as xgb

NUMBER_ROUNDS = 90
EARLY_STOPPING_ROUNDS = 5


X_START_IDX = 6
Y_IDX = 2
TRAIN_BUFFER_FILE = "data/train_rank_date_high.buffer"
VALI_BUFFER_FILE = "data/vali_rank_date_high.buffer"

def get_buffer_file(data_type="train",y_idx=2):
    return "data/rank_%s_%d.buffer"%(data_type,y_idx)

def get_group_info(data_type):
    with open('data/all_sample_group_%s.txt'%(data_type),'r') as f:
        return list(f)

def get_group_info_2(data):
    li = []
    val = None 
    val_count = 0
    for x in data:
        if x!=val:
            if val:
                li.append(val_count)
            val = x
            val_count = 1
        else:
            val_count = val_count + 1
    li.append(val_count)
    # print(li)
    return li 

def prepare_data(y_idx=2):
    train_data = np.loadtxt('data/all_sample_train.txt', delimiter=',')
    train_X = np.ascontiguousarray(train_data[:, X_START_IDX:])
    train_Y = np.ascontiguousarray(train_data[:, y_idx])  # 2-最高价格，3-最低价格

    vali_data = np.loadtxt('data/all_validate.txt', delimiter=',')
    vali_X = np.ascontiguousarray(vali_data[:, X_START_IDX:])
    vali_Y = np.ascontiguousarray(vali_data[:, y_idx])

    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_vali = xgb.DMatrix(vali_X, label=vali_Y)

    # GROUP_ITEMS_COUNT = 20
    group_train = get_group_info("train")
    # group_valid = get_group_info("validate")
    group_valid = get_group_info_2(vali_data[:,0])
    
    xg_train.set_group(group_train)
    xg_vali.set_group(group_valid)

    xg_train.save_binary(get_buffer_file("train",y_idx)) 
    xg_vali.save_binary(get_buffer_file("vali",y_idx))
    return xg_train, xg_vali


def train(y_idx=2):
    xg_train, xg_vali = (None, None)
    ftrain = get_buffer_file("train",y_idx)
    fvali = get_buffer_file("vali",y_idx)
    if os.path.exists(ftrain) and os.path.exists(fvali):
        print("read from cache")
        xg_train = xgb.DMatrix(ftrain)
        xg_vali = xgb.DMatrix(fvali)
    else:
        xg_train, xg_vali = prepare_data(y_idx)

    watchlist = [(xg_train, 'train'), (xg_vali, 'vali')]

    print("begin pairwise")
    param = {'objective': 'rank:pairwise',
             'eval_metric': 'auc',
             'eta': 0.3, 'gamma': 1,
             'min_child_weight': 0.05,
             'max_depth': 6}
    bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    bst.save_model("model_tmp/pairwise_%d_model.json"%(y_idx))

    print("begin listwise")
    param = {'objective': 'rank:ndcg',
             'eval_metric': 'ndcg',
             'eta': 0.3, 'gamma': 1,
             'min_child_weight': 0.05,
             'max_depth': 6}
    bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    bst.save_model("model_tmp/listwise_%d_model.json"%(y_idx))



def test():
    data = np.loadtxt('data/all_train.txt', delimiter=',')
    print(len(data[:,0]))
    a = get_group_info(data[:,0])
    print(a)

    # data = np.loadtxt('data/all_validate.txt', delimiter=',')
    # a = get_group_info(data[:0])
    # print(a)
        
# python train_rank_date.py 2   # y-high
# python train_rank_date.py 3   # y-low
# python train_rank_date.py 4   # label-high
# python train_rank_date.py 5   # label-low
if __name__ == "__main__":
    y_idx = int(sys.argv[1])
    assert y_idx in [2,3,4,5] , "y_idx not in [2,3,4,5]"
    # print(y_idx,sys.argv[1])
    train(y_idx)
    # test()

