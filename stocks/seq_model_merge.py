import os
import sys
from typing import Optional
import numpy as np
import pandas as pd
import json
import sqlite3
import math
from common import c_round, load_trade_dates


def model_intersection():
    df_predict_v1 = pd.read_csv(
        "data/predict/predict_merged.txt", sep=";", header=0, dtype={"stock_no": str}
    )
    sel_fields = "pk_date_stock,stock_no,pair_15,top3".split(",")
    df_predict_v1 = df_predict_v1[sel_fields]
    # df_predict_v1 = df_predict_v1.drop("point_low1,point_high1,buy_prices,sell_prices".split(","),axis=1)
    df_predict_v1 = df_predict_v1.rename(columns={"top3": "top3_v1"})

    df_predict_v2 = pd.read_csv(
        "data/predict_v2/predict_merged.txt", sep=";", header=0, dtype={"stock_no": str}
    )
    df_predict_v1 = df_predict_v1.drop(["pk_date_stock"], axis=1)
    # sel_fields = "stock_no,top3,list_dates,point_low1,point_high1,buy_prices,sell_prices".split(",")
    # df_predict_v2 = df_predict_v2[sel_fields]
    df_predict_v2 = df_predict_v2.rename(columns={"top3": "top3_v2"})

    df_predict_v1 = df_predict_v1.merge(df_predict_v2, on="stock_no", how="left")
    df_predict_v1["top3"] = df_predict_v1["top3_v1"] + df_predict_v1["top3_v2"]

    # pair_15,list_dates
    df_predict_v1 = df_predict_v1.sort_values(
        by=["point2pair_dates_top5", "list_dates_top5"], ascending=False
    )
    df_predict_v1.to_csv(
        "data/predict_v2/predict_merged_v1_2.txt", sep=";", index=False
    )

    # # 只关注预测结果top3=3部分的数据
    # df_predict_v2 = df_predict_v2[df_predict_v2['top3']==3]
    # df_predict_v2['lowest_rate'] = df_predict_v2['point_low1']


def statics(df, trade_date, orderby, topN):
    df = df.sort_values(by=["top3", orderby], ascending=False)  #
    df_topN = df.head(topN)
    li = [trade_date, orderby, topN]
    for i in range(5):
        tmpdf = df_topN[df_topN[f"high2_rate_{i}"] != 0]
        success_rate = c_round(len(tmpdf) / topN)
        mean_earn_rate = 0
        real_earn_rate = 0
        if len(tmpdf) > 0:
            mean_earn_rate = c_round(tmpdf[f"high2_rate_{i}"].mean())
            real_earn_rate = c_round(mean_earn_rate * success_rate)
        li = li + [success_rate, mean_earn_rate, real_earn_rate]
    return li


def merge_one(trade_date, df_predict, df_real, version="v2"):
    filename = f"data/predict_{version}/predict_true_{trade_date}.txt"
    print(filename)
    
    df_predict = df_predict.merge(df_real, on="stock_no", how="left")

    df_predict["low_rate"] = c_round(
        (df_predict["LOW_1"] - df_predict["CLOSE_price"]) / df_predict["CLOSE_price"]
    )
    df_predict["high2_rate"] = c_round(
        (df_predict["high_2"] - df_predict["CLOSE_price"]) / df_predict["CLOSE_price"]
    )
    df_predict["high3_rate"] = c_round(
        (df_predict["high_3"] - df_predict["CLOSE_price"]) / df_predict["CLOSE_price"]
    )

    for i in range(5):
        df_predict[f"point_low1_buy_{i}"] = df_predict.apply(
            lambda x: 0 if x["low_rate"] >= x[f"point_low1_{i}"] else 1, axis=1
        )
        df_predict[f"high2_rate_{i}"] = df_predict.apply(
            lambda x: 0
            if x["low_rate"] >= x[f"point_low1_{i}"]
            else (x["high2_rate"] - x[f"point_low1_{i}"]),
            axis=1,
        )
        df_predict[f"high3_rate_{i}"] = df_predict.apply(
            lambda x: 0
            if x["low_rate"] >= x[f"point_low1_{i}"]
            else (x["high3_rate"] - x[f"point_low1_{i}"]),
            axis=1,
        )

    # point_low1_options = [-0.0299, -0.0158, 0, 0.0193, 0.025]
    # df_predict['buy_success_count'] = df_predict.apply(lambda x: sum([0 if x['low_rate']>=(x['point_low1'] + t) else 1  for t in point_low1_options]) , axis=1)

    df_predict.to_csv(filename, sep=";", index=False)
    

    statics_li = []
    order_fields = "list_dates,point,point2pair_dates,point_high1".split(",")
    for field in order_fields:
        for topn in [5, 10]:
            ret = statics(df_predict, trade_date, field, topn)
            statics_li.append(ret)
    return statics_li


def merge_predict_true(start_date=20231017):
    conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)
    sql = f"select distinct trade_date from stock_raw_daily_2 where trade_date>{start_date}"
    trade_dates = (
        pd.read_sql(sql, conn)["trade_date"].sort_values(ascending=True).tolist()
    )
    count = len(trade_dates)

    statics_li = []
    for idx, trade_date in enumerate(trade_dates):
        print(trade_date)
        if idx + 2 >= count:
            break

        # 第一天，昨收，开盘价，最低价
        df_real = pd.read_sql(
            f"select stock_no,OPEN_price as OPEN_1,LOW_price as LOW_1 from stock_raw_daily_2 where trade_date={trade_dates[idx]}",
            conn,
        )

        # 第二天，最高价
        df_real = df_real.merge(
            pd.read_sql(
                f"select stock_no,HIGH_price as high_2 from stock_raw_daily_2 where trade_date={trade_dates[idx+1]}",
                conn,), on="stock_no", how="left",
        )

        # 第三天，最高价
        df_real = df_real.merge(
            pd.read_sql(
                f"select stock_no,HIGH_price as high_3 from stock_raw_daily_2 where trade_date={trade_dates[idx+2]}",
                conn,
            ), on="stock_no", how="left",
        )

        # df_predict_v1 = pd.read_csv(f"data/predict/predict_merged.txt.{trade_date}",sep=";",header=0,dtype={'stock_no': str})
        # df_predict_v1 = df_predict_v1.rename(columns={'low1.7':'point_low1'})
        # merge_one(trade_date,df_predict_v1,df_real,'v1')

        df_predict_v2 = pd.read_csv(
            f"data/predict_v2/predict_merged.txt.{trade_date}",
            sep=";",
            header=0,
            dtype={"stock_no": str},
        )
        statics_li = statics_li + merge_one(trade_date, df_predict_v2, df_real, "v2")
        # if idx > 1:
        #     break

    # success_rate,mean_earn_rate,real_earn_rate
    columns = "trade_date,order_by,topN".split(",")
    for i in range(5):
        columns.append(f"succ{i}")
        columns.append(f"earn{i}")
        columns.append(f"Rearn{i}")
    df = pd.DataFrame(statics_li, columns=columns)
    filename = f"predict_true_static_{trade_date}.txt"
    print(filename)
    df.to_csv(filename, sep=";", index=False)


# python seq_model_merge.py intersection
if __name__ == "__main__":
    op_type = sys.argv[1]
    if op_type == "intersection":
        model_intersection()
    if op_type == "predict_true":
        merge_predict_true()
