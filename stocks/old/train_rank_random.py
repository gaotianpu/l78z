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
Y_IDX = 2
TRAIN_BUFFER_FILE = "data/train_rank_random_high.buffer"
VALI_BUFFER_FILE = "data/vali_rank_random_high.buffer"


def get_group_info(sample_count, group_item_count=20):
    group_count = sample_count//group_item_count
    gc_mod = sample_count % group_item_count
    l = [group_item_count for i in range(group_count)]
    if gc_mod > 0:
        l.append(gc_mod)
    return l


def prepare_data():
    x_train, y_train = load_svmlight_file(
        "data/high_train_rank.txt")
    x_valid, y_valid = load_svmlight_file(
        "data/high_validate_rank.txt")

    train_dmatrix = xgb.DMatrix(x_train, y_train)
    valid_dmatrix = xgb.DMatrix(x_valid, y_valid)
    # test_dmatrix = xgb.DMatrix(x_test)

    GROUP_ITEMS_COUNT = 20
    group_train = get_group_info(train_dmatrix.num_row(), GROUP_ITEMS_COUNT)
    group_valid = get_group_info(valid_dmatrix.num_row(), GROUP_ITEMS_COUNT)
    train_dmatrix.set_group(group_train)
    valid_dmatrix.set_group(group_valid)

    train_dmatrix.save_binary(TRAIN_BUFFER_FILE)
    valid_dmatrix.save_binary(VALI_BUFFER_FILE)
    return train_dmatrix, valid_dmatrix


def train():
    xg_train, xg_vali = (None, None)
    if os.path.exists(TRAIN_BUFFER_FILE) and os.path.exists(TRAIN_BUFFER_FILE):
        xg_train = xgb.DMatrix(TRAIN_BUFFER_FILE)
        xg_vali = xgb.DMatrix(VALI_BUFFER_FILE)
    else:
        xg_train, xg_vali = prepare_data()

    param = {'objective': 'rank:pairwise',
             'eval_metric': 'auc',
             'eta': 0.2, 'gamma': 1,
             'min_child_weight': 0.05,
             'max_depth': 6}

    watchlist = [(xg_train, 'train'), (xg_vali, 'vali')]
    bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    bst.save_model("model/train_rank_random.json")


if __name__ == "__main__":
    train()
