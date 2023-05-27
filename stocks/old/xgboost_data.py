#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.datasets import load_svmlight_file
import xgboost as xgb

xgb.set_config(verbosity=2)


def regression_train(predict_type="high"):
    TRAIN_BUFFER_FILE = "data/%s_train.buffer" % (predict_type)
    VALI_BUFFER_FILE = "data/%s_vali.buffer" % (predict_type)

    y_idx = 2
    if predict_type == "low":
        y_idx = 3
    if predict_type == "high_label":
        y_idx = 4
    if predict_type == "updown_label":
        y_idx = 5

    train_data = np.loadtxt('data/train.txt', delimiter=',')
    train_X = np.ascontiguousarray(train_data[:, 6:])
    train_Y = np.ascontiguousarray(train_data[:, y_idx])  # 2-最高价格，3-最低价格

    vali_data = np.loadtxt('data/validate.txt', delimiter=',')
    vali_X = np.ascontiguousarray(vali_data[:, 6:])
    vali_Y = np.ascontiguousarray(vali_data[:, y_idx])

    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_vali = xgb.DMatrix(vali_X, label=vali_Y)

    xg_train.save_binary(TRAIN_BUFFER_FILE)
    xg_vali.save_binary(VALI_BUFFER_FILE)


def get_group_info(sample_count, group_item_count=20):
    group_count = sample_count//group_item_count
    gc_mod = sample_count % group_item_count
    l = [group_item_count for i in range(group_count)]
    if gc_mod > 0:
        l.append(gc_mod)
    return l


def group_rank_data(predict_type="high"):
    TRAIN_BUFFER_FILE = "data/grouprank_%s_train.buffer" % (predict_type)
    VALI_BUFFER_FILE = "data/grouprank_%s_vali.buffer" % (predict_type)

    x_train, y_train = load_svmlight_file(
        "data/%s_train_rank.txt" % (predict_type))
    x_valid, y_valid = load_svmlight_file(
        "data/%s_validate_rank.txt" % (predict_type))
    # x_test, y_test = load_svmlight_file("data/v3/rank_%s_test.txt"% (predict_type))

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

def group_rank_data_by_date(predict_type="high"):
    TRAIN_BUFFER_FILE = "data/grouprank_%s_train_date.buffer" % (predict_type) 

    x_train, y_train = load_svmlight_file(
        "data/%s_train_rank_date.txt" % (predict_type)) 

    train_dmatrix = xgb.DMatrix(x_train, y_train) 

    GROUP_ITEMS_COUNT = 20
    group_train = get_group_info(train_dmatrix.num_row(), GROUP_ITEMS_COUNT) 
    train_dmatrix.set_group(group_train) 

    train_dmatrix.save_binary(TRAIN_BUFFER_FILE)

def process():
    regression_train("high")
    # regression_train("low")

    # regression_train("high_label")
    regression_train("updown_label")

    group_rank_data("high")
    # group_rank_data("low")

    group_rank_data_by_date('high')
    # group_rank_data_by_date('low')


if __name__ == "__main__":
    process()
