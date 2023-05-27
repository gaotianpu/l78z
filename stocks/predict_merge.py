#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd 

# predict/predict_scores_today.csv
# 
# 

TOP_N = 80
FIELDS = "num,STOCK,pointwise_4_model,pointwise_3_model,pairwise_4_model,pairwise_5_model,listwise_4_model,listwise_5_model".split(",")

def get_predict_date():
    with  open("data/predict.txt",'r') as f:
        for line in f:
            return line.split(",")[0]


def get_rnn_point_predict_scores():
    with open('predict/rnn_predict_scores.txt','r') as f:
        d = {}
        for i,line in enumerate(f):
            parts = line.strip().split(",")
            if parts[1] != "nan":
                d[parts[0]] = float(parts[1])
        return d

def get_rnn_pair_predict_scores():
    with open('predict/rnn_predict_pair_scores.txt','r') as f:
        d = {}
        for i,line in enumerate(f):
            parts = line.strip().split(",")
            if parts[1] != "nan":
                d[parts[0]] = float(parts[1])
        return d

def merge_predict_scores():
    scores = get_rnn_point_predict_scores()
    pair_scores = get_rnn_pair_predict_scores()

    df = pd.read_csv("predict/predict_scores_today.csv", dtype={'STOCK': str})  # 指定stock_no列为字符串类型
    # print(df)
    df.insert(2, "rnn_point", 0.0)
    for index, row in df.iterrows():
        df.loc[index, "rnn_point"] = scores.get(row["STOCK"], 0.0)
    
    df.insert(2, "rnn_pair", 0.0)
    for index, row in df.iterrows():
        df.loc[index, "rnn_pair"] = pair_scores.get(row["STOCK"], 0.0)

    topn = df.sort_values(by="rnn_point", ascending=False).head(TOP_N)["STOCK"].tolist()
    for index, row in df.iterrows():
        if row["STOCK"] in topn:
            df.loc[index, "num"] = df.loc[index, "num"] + 1 
    
    topn = df.sort_values(by="rnn_pair", ascending=False).head(TOP_N)["STOCK"].tolist()
    for index, row in df.iterrows():
        if row["STOCK"] in topn:
            df.loc[index, "num"] = df.loc[index, "num"] + 1 

    df = df.sort_values(
        by=["num", "pairwise_4_model"], ascending=[False, False]
    )

    predict_date = get_predict_date()
    df.to_csv("predict/predict_scores_final_%s.csv" % (predict_date), index=False)
    df.to_csv("predict/predict_scores_final_today.csv" , index=False)

    # df.to_csv("predict/predict_scores_x.csv", index=False)


if __name__ == "__main__":
    merge_predict_scores()

