#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
https://www.jianshu.com/p/2920c97e9e16
"""
import sys
import numpy as np
import pandas as pd
from collections import Counter
import xgboost as xgb

X_START_IDX = 6

MODEL_NAMES = "pointwise_4_model,pointwise_3_model,pairwise_4_model,pairwise_5_model,listwise_4_model,listwise_5_model".split(
    ","
)


def pred(model_name, x_data, model_release_path="model"):
    model_file = "%s/%s.json" % (model_release_path,model_name)
    # print(model_file)
    # return []
    bst = xgb.Booster()
    bst.load_model(model_file)
    pred_y = bst.predict(x_data)
    return pred_y


def process(model_release_path="model"):
    data_file = "data/predict.txt"
    df = pd.read_csv(data_file, dtype={1: str})  # 指定stock_no列为字符串类型
    trade_date = df.iloc[:, 0][0]
    stock_no = df.iloc[:, 1]
    xgb_X = xgb.DMatrix(np.ascontiguousarray(df.iloc[:, X_START_IDX:]))

    li = []
    for model in MODEL_NAMES:
        li.append(pred(model, xgb_X,model_release_path))

    scores = list(zip(stock_no, li[0], li[1], li[2], li[3], li[4], li[5]))
    df_scores = pd.DataFrame(scores, columns=["STOCK"] + MODEL_NAMES)

    TOP_N = 80
    topn_stocks = []
    topn_stocks = (
        topn_stocks
        + df_scores.sort_values(by="pointwise_4_model", ascending=False)
        .head(TOP_N)["STOCK"]
        .tolist()
    )
    topn_stocks = (
        topn_stocks
        + df_scores.sort_values(by="pairwise_4_model", ascending=False)
        .head(TOP_N)["STOCK"]
        .tolist()
    )
    topn_stocks = (
        topn_stocks
        + df_scores.sort_values(by="listwise_4_model", ascending=False)
        .head(TOP_N)["STOCK"]
        .tolist()
    )
    stocks_cnt = dict(Counter(topn_stocks))

    df_scores.insert(1, "num", 0)
    for index, row in df_scores.iterrows():
        df_scores.loc[index, "num"] = stocks_cnt.get(row["STOCK"], 0)

    df_scores = df_scores.sort_values(
        by=["num", "pairwise_4_model"], ascending=[False, False]
    )
    df_scores.to_csv("predict/predict_scores_%d.csv" % (trade_date), index=False)
    df_scores.to_csv("predict/predict_scores_today.csv" , index=False)
    # print(scores)


if __name__ == "__main__":
    model_release_path = sys.argv[1] if sys.argv[1] else "model"
    process(model_release_path)
