#!/usr/bin/python
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
import xgboost as xgb


def r_rank_train(predict_type="high", data_cache=True):
    TRAIN_BUFFER_FILE = "data/r_rank_%s_train.buffer" % (predict_type)
    VALI_BUFFER_FILE = "data/r_rank_%s_vali.buffer" % (predict_type)

    xg_train, xg_vali = (None, None)
    if data_cache and os.path.exists(TRAIN_BUFFER_FILE):
        xg_train = xgb.DMatrix(TRAIN_BUFFER_FILE)
        xg_vali = xgb.DMatrix(VALI_BUFFER_FILE)
    else:
        y_idx = 2
        if predict_type == "mean":
            y_idx = 3

        train_data = np.loadtxt('data/v3/train3.txt', delimiter=',')
        train_X = np.ascontiguousarray(train_data[:, 4:])
        train_Y = np.ascontiguousarray(train_data[:, y_idx])  # 2-最高价格，3-最低价格

        vali_data = np.loadtxt('data/v3/validate.txt', delimiter=',')
        vali_X = np.ascontiguousarray(vali_data[:, 4:])
        vali_Y = np.ascontiguousarray(vali_data[:, y_idx])

        GROUP_ITEMS_COUNT = 20
        group_train = get_group_info(train_Y.shape[0], GROUP_ITEMS_COUNT)
        group_valid = get_group_info(vali_Y.shape[0], GROUP_ITEMS_COUNT)

        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_vali = xgb.DMatrix(vali_X, label=vali_Y)

        xg_train.set_group(group_train)
        xg_vali.set_group(group_valid)

        xg_train.save_binary(TRAIN_BUFFER_FILE)
        xg_vali.save_binary(VALI_BUFFER_FILE)

    # save_bin?

    num_round = 7

    # setup parameters for xgboost
    # https://www.biaodianfu.com/xgboost.html
    # 'rank:pairwise'
    # 'rank:ndcg'
    param = {'objective': 'rank:ndcg',
             'eval_metric': 'ndcg@2',
             'eta': 0.08, 'gamma': 1,
             'min_child_weight': 0.05,
             'max_depth': 8,
             'nthread': 4}

    watchlist = [(xg_train, 'train'), (xg_vali, 'vali')]
    bst = xgb.train(param, xg_train, num_round, watchlist)
    bst.save_model("model/r_rank_%s_model.json" % (predict_type))
    # early_stopping_rounds ?


def rank_ndcg_train(predict_type="high", data_cache=True):
    TRAIN_BUFFER_FILE = "data/rank_%s_train.buffer" % (predict_type)
    VALI_BUFFER_FILE = "data/rank_%s_vali.buffer" % (predict_type)

    xg_train, xg_vali = (None, None)
    if data_cache and os.path.exists(TRAIN_BUFFER_FILE):
        xg_train = xgb.DMatrix(TRAIN_BUFFER_FILE)
        xg_vali = xgb.DMatrix(VALI_BUFFER_FILE)
    else:
        y_idx = 2
        if predict_type == "mean":
            y_idx = 3
        train_data = np.loadtxt('data/v2/train.txt', delimiter=',')
        train_X = np.ascontiguousarray(train_data[:, 4:])
        train_Y = np.ascontiguousarray(train_data[:, y_idx])  # 2-最高价格，3-最低价格

        vali_data = np.loadtxt('data/v2/validate.txt', delimiter=',')
        vali_X = np.ascontiguousarray(vali_data[:, 4:])
        vali_Y = np.ascontiguousarray(vali_data[:, y_idx])

        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_vali = xgb.DMatrix(vali_X, label=vali_Y)

        GROUP_ITEMS_COUNT = 20
        group_train = get_group_info(train_Y.shape[0], GROUP_ITEMS_COUNT)
        group_valid = get_group_info(vali_Y.shape[0], GROUP_ITEMS_COUNT)
        xg_train.set_group(group_train)
        xg_vali.set_group(group_valid)

        xg_train.save_binary(TRAIN_BUFFER_FILE)
        xg_vali.save_binary(VALI_BUFFER_FILE)

    # setup parameters for xgboost
    # https://www.biaodianfu.com/xgboost.html
    # 'rank:pairwise'
    # 'rank:ndcg'
    param = {'objective': 'rank:ndcg',
             'eval_metric': 'ndcg@2',
             'eta': 0.05, 'gamma': 1.0,
             'min_child_weight': 0.05,
             'max_depth': 8,
             'nthread': 4}

    watchlist = [(xg_train, 'train'), (xg_vali, 'vali')]
    bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    bst.save_model("model/rank_ndcg_%s_model.json" % (predict_type))


def regression_rank_pairewise_train(predict_type="high"):
    TRAIN_BUFFER_FILE = "data/regression_%s_train.buffer" % (predict_type)
    VALI_BUFFER_FILE = "data/regression_%s_vali.buffer" % (predict_type)

    xg_train = xgb.DMatrix(TRAIN_BUFFER_FILE)
    xg_vali = xgb.DMatrix(VALI_BUFFER_FILE)

    # print(dir(xg_train),xg_train.num_row())

    GROUP_ITEMS_COUNT = 20
    group_train = get_group_info(xg_train.num_row(), GROUP_ITEMS_COUNT)
    group_valid = get_group_info(xg_vali.num_row(), GROUP_ITEMS_COUNT)
    xg_train.set_group(group_train)
    xg_vali.set_group(group_valid)

    # setup parameters for xgboost
    # https://www.biaodianfu.com/xgboost.html
    # 'rank:pairwise'
    # 'rank:ndcg'
    # 'eval_metric': 'ndcg@3',
    param = {'objective': 'rank:pairwise',
             #  'eval_metric':'auc',
             'eta': 0.05, 'gamma': 0.1,
             'min_child_weight': 0.05,
             'subsample': 0.99,
             'max_depth': 8,
             'nthread': 8}

    watchlist = [(xg_train, 'train'), (xg_vali, 'vali')]
    bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    bst.save_model("model/regression_rank_pairewise_%s_model.json" %
                   (predict_type))

    # high [17]	train-auc:0.57720	vali-auc:0.48111
    # low  [22]	train-auc:0.60337	vali-auc:0.54920

    # high train-auc:0.57132	vali-auc:0.47915
    # low  train-auc:0.59368	vali-auc:0.54012

    # [3]	train-map:0.83921	vali-map:0.94693
    # [6]   train-map:0.84866	vali-map:0.95137


def rank_pairewise_train(predict_type="high", data_cache=True):
    TRAIN_BUFFER_FILE = "data/rank_%s_train.buffer" % (predict_type)
    VALI_BUFFER_FILE = "data/rank_%s_vali.buffer" % (predict_type)

    xg_train, xg_vali = (None, None)
    if data_cache and os.path.exists(TRAIN_BUFFER_FILE):
        xg_train = xgb.DMatrix(TRAIN_BUFFER_FILE)
        xg_vali = xgb.DMatrix(VALI_BUFFER_FILE)
    else:
        y_idx = 2
        if predict_type == "mean":
            y_idx = 3
        train_data = np.loadtxt('data/v2/train.txt', delimiter=',')
        train_X = np.ascontiguousarray(train_data[:, 4:])
        train_Y = np.ascontiguousarray(train_data[:, y_idx])  # 2-最高价格，3-最低价格

        vali_data = np.loadtxt('data/v2/validate.txt', delimiter=',')
        vali_X = np.ascontiguousarray(vali_data[:, 4:])
        vali_Y = np.ascontiguousarray(vali_data[:, y_idx])

        GROUP_ITEMS_COUNT = 20
        group_train = get_group_info(train_Y.shape[0], GROUP_ITEMS_COUNT)
        group_valid = get_group_info(vali_Y.shape[0], GROUP_ITEMS_COUNT)

        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_vali = xgb.DMatrix(vali_X, label=vali_Y)

        xg_train.set_group(group_train)
        xg_vali.set_group(group_valid)

        xg_train.save_binary(TRAIN_BUFFER_FILE)
        xg_vali.save_binary(VALI_BUFFER_FILE)

    # setup parameters for xgboost
    # https://www.biaodianfu.com/xgboost.html
    # 'rank:pairwise'
    # 'rank:ndcg'
    # 'eval_metric': 'ndcg@3',
    param = {'objective': 'rank:pairwise',
             'eta': 0.05, 'gamma': 1.0,
             'min_child_weight': 0.05,
             'max_depth': 8,
             'nthread': 4}

    watchlist = [(xg_train, 'train'), (xg_vali, 'vali')]
    bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    bst.save_model("model/rank_pairewise_%s_model.json" % (predict_type))


def multiclass_classification_train(predict_type="high_label"):
    TRAIN_BUFFER_FILE = "data/regression_%s_train.buffer" % (predict_type)
    VALI_BUFFER_FILE = "data/regression_%s_vali.buffer" % (predict_type)

    xg_train = xgb.DMatrix(TRAIN_BUFFER_FILE)
    xg_vali = xgb.DMatrix(VALI_BUFFER_FILE)

    param = {
        'objective': 'multi:softmax',
        'num_class': 5,
        'eval_metric': 'auc',  # auc ? #mlogloss
        'eta': 0.1,
        'gamma': 0.2,
        'lambda': 2,
        'max_depth': 6,
        'nthread': 8
    }

    watchlist = [(xg_train, 'train'), (xg_vali, 'vali')]
    bst = xgb.train(param, xg_train, NUMBER_ROUNDS, watchlist,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    bst.save_model("model/multiclass_classification_%s_model.json" %
                   (predict_type))

    # high [27]	train-mlogloss:1.49599	vali-mlogloss:1.55662
    # low  [4]	train-mlogloss:1.55128	vali-mlogloss:1.60644

    # high [3]	train-auc:0.60899	vali-auc:0.60762
    # low  [18]	train-auc:0.63857	vali-auc:0.59118


def test_1():
    x_test, y_test, qid = load_svmlight_file(
        "svmlight_file_test.txt", multilabel=False, query_id=True)
    print(x_test)
    print("--")
    print(y_test)
    print("--")
    print(qid)


def get_data_describe():
    # high: 75%=0.37,50%=0.14,25%=
    df = pd.read_csv("train.high_low.txt", delimiter=" ",
                     names=['high', 'low'])
    desc = df.describe()
    print("all==:")
    for key, r in desc.iterrows():
        # print("i:",i)
        print(key, r[0], r[1])
    # print(type(desc) )
    # print(df.describe())

    df_high = df[df.high > 0.037]
    # print(df_high.describe())
    print("high==:")
    for key, r in df_high.describe().iterrows():
        # print("i:",i)
        print(key, r[0], r[1])

    df_low = df[df.low > 0.0]
    print("low==:")
    for key, r in df_low.describe().iterrows():
        # print("i:",i)
        print(key, r[0], r[1])

    # high 0.062  0.037  0.014  -0.002
    # min  0.012  0.0, -0.015  -0.034

    # count 1913723.0 1913723.0
    # mean 0.020979509051188026 -0.01806445081129964
    # std 0.047790039028556004 0.04171726468895279
    # min -1.0 -1.0
    # 25% -0.002 -0.034
    # 50% 0.014 -0.015
    # 75% 0.037 0.0
    # max 0.559 0.515

# cat data/v1/test.txt 2 | python test.py > high.text.svmf
# cat data/v1/test.txt 3 | python test.py > low.text.svmf
# cat data/v1/test.txt 4 | python test.py > high.label.text.svmf
# cat data/v1/test.txt 5 | python test.py > low.label.text.svmf


def convert_csv2svm_line(line, y_idx):
    fields = line.strip().split(",")

    svm_fields = ["%d:%s" % (f_idx+1, f)
                  for f_idx, f in enumerate(fields[6:])]
    svm_fields.insert(0, "qid:%s%s" % (fields[0], fields[1]))
    svm_fields.insert(0, fields[y_idx])

    print(" ".join(svm_fields))


def convert_csv_to_svm(y_idx=2):
    for i, line in enumerate(sys.stdin):
        convert_csv2svm_line(line, y_idx)

        # fields = line.strip().split(",")

        # svm_fields = ["%d:%s" % (f_idx+1, f)
        #               for f_idx, f in enumerate(fields[6:])]
        # svm_fields.insert(0, "qid:%s%s" % (fields[0], fields[1]))
        # svm_fields.insert(0, fields[y_idx])

        # print(" ".join(svm_fields))


if __name__ == "__main__":
    y_idx = sys.argv[1]
    print(y_idx)
    convert_csv_to_svm()
    # get_data_describe()
