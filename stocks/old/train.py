#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import sklearn
from sklearn.datasets import load_svmlight_file
import xgboost as xgb

NUMBER_ROUNDS = 80
EARLY_STOPPING_ROUNDS = 3


def regression_train(predict_type="high"):
    print("regression_train_%s" %(predict_type) )
    TRAIN_BUFFER_FILE = "data/%s_train.buffer" % (predict_type)
    VALI_BUFFER_FILE = "data/%s_vali.buffer" % (predict_type)

    xg_train = xgb.DMatrix(TRAIN_BUFFER_FILE)
    xg_vali = xgb.DMatrix(VALI_BUFFER_FILE)

    # setup parameters for xgboost
    # https://www.biaodianfu.com/xgboost.html
    param = {
        'objective': 'reg:squarederror',
        'eta': 0.08,
        'gamma': 0.2,
        'lambda': 2,
        'verbosity':1, # 0 (silent), 1 (warning), 2 (info), 3 (debug)
        'max_depth': 8
    }

    watchlist = [(xg_train, 'train'), (xg_vali, 'vali')]
    bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    bst.save_model("model/regression_%s_model.json" % (predict_type))


def multiclass_train():
    print("multiclass_train")
    TRAIN_BUFFER_FILE = "data/updown_label_train.buffer"  
    VALI_BUFFER_FILE = "data/updown_label_vali.buffer"  

    xg_train = xgb.DMatrix(TRAIN_BUFFER_FILE)
    xg_vali = xgb.DMatrix(VALI_BUFFER_FILE)

    # setup parameters for xgboost
    # https://www.biaodianfu.com/xgboost.html
    # Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softmax' was changed 
    # from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    param = {
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'num_class': 4,
        'eta': 0.08,
        'gamma': 0.2,
        'lambda': 2,
        'verbosity': 1,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
        'max_depth': 8
    }

    watchlist = [(xg_train, 'train'), (xg_vali, 'vali')]
    bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    bst.save_model("model/multiclass_model.json")

def get_group_info(sample_count, group_item_count=20):
    group_count = sample_count//group_item_count
    gc_mod = sample_count % group_item_count
    l = [group_item_count for i in range(group_count)]
    if gc_mod > 0:
        l.append(gc_mod)
    return l


def rank_train(predict_type="high"):
    print("rank_train_%s" %(predict_type) )
    TRAIN_BUFFER_FILE = "data/grouprank_%s_train.buffer" % (predict_type)
    VALI_BUFFER_FILE = "data/grouprank_%s_vali.buffer" % (predict_type)

    xg_train = xgb.DMatrix(TRAIN_BUFFER_FILE)
    xg_vali = xgb.DMatrix(VALI_BUFFER_FILE)

    param = {'objective': 'rank:ndcg',
             'eval_metric': 'ndcg@2',
             'eta': 0.08, 'gamma': 1,
             'min_child_weight': 0.05,
             'max_depth': 8}

    watchlist = [(xg_train, 'train'), (xg_vali, 'vali')]
    bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    bst.save_model("model/rank_%s_model.json" % (predict_type))

def rank_train_by_date(predict_type="high"): 
    TRAIN_BUFFER_FILE = "data/grouprank_%s_train_date.buffer" % (predict_type)
    VALI_BUFFER_FILE = "data/grouprank_%s_vali.buffer" % (predict_type)

    xg_train = xgb.DMatrix(TRAIN_BUFFER_FILE)
    xg_vali = xgb.DMatrix(VALI_BUFFER_FILE)
    watchlist = [(xg_train, 'train'), (xg_vali, 'vali')] 

    print("rank_train_by_date_pairwise_%s" %(predict_type) )
    param = {'objective': 'rank:pairwise',
             'eval_metric': 'auc',
             'eta': 0.08, 'gamma': 1,
             'min_child_weight': 0.05,
             'max_depth': 6}  
    bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    bst.save_model("model/rank_%s_model_date_pairwise.json" % (predict_type))

    print("rank_train_by_date_ndcg_%s" %(predict_type) )
    param = {'objective': 'rank:ndcg',
             'eta': 0.08, 'gamma': 1,
             'min_child_weight': 0.05,
             'max_depth': 6}  
    bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    bst.save_model("model/rank_%s_model_date_ndcg.json" % (predict_type)) 

    #high train-ndcg@2:0.48275    vali-ndcg@2:0.41089
    #low  train-ndcg@2:0.34273    vali-ndcg@2:0.34813

def process():
    regression_train("high")
    multiclass_train()
    rank_train("high")
    rank_train_by_date("high")

    

    # regression_train("low") 
    # rank_train("low") 
    # rank_train_by_date("low")


if __name__ == "__main__":
    # tmp("high")
    process()
