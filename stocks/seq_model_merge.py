import os
import sys
import glob
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
        by=["top3", "point_high1"], ascending=False
    )
    df_predict_v1.to_csv(
        "data/predict_v2/predict_merged_v1_2.txt", sep=";", index=False
    )

    # # 只关注预测结果top3=3部分的数据
    # df_predict_v2 = df_predict_v2[df_predict_v2['top3']==3]
    # df_predict_v2['lowest_rate'] = df_predict_v2['point_low1']


# def buy_statics(df, trade_date, orderby, topN):
#     df = df.sort_values(by=["top3", orderby], ascending=False)  #
#     df_topN = df.head(topN)
#     li = [trade_date, orderby, topN]
#     for i in range(5):
#         tmpdf = df_topN[df_topN[f"point_low1_buy_{i}"] != 0] #
#         success_rate = c_round(len(tmpdf) / topN)
#         mean_earn_rate = 0
#         real_earn_rate = 0
#         if len(tmpdf) > 0:
#             mean_earn_rate = c_round(tmpdf[f"high2_rate_{i}"].mean())
#             real_earn_rate = c_round(mean_earn_rate * success_rate)
#         li = li + [success_rate, mean_earn_rate, real_earn_rate]
#     return li

# def sell_statics(df, trade_date, orderby, topN):
#     df = df.sort_values(by=["top3", orderby], ascending=False)  #
#     df_topN = df.head(topN)
#     li = [trade_date, orderby, topN]
#     for i in range(5):
#         tmpdf = df_topN[df_topN[f"point_high1_sell_{i}"] != 0]
#         success_rate = c_round(len(tmpdf) / topN)
#         li.append(success_rate)
#     return li

def merge_one(trade_date, df_predict, df_real, version="v2"):
    filename = f"data/predict_{version}/predict_true_{trade_date}.txt"
    print(filename)
    # rm -f data/predict_v2/predict_true_20231*
    
    if os.path.exists(filename):
        df_predict = pd.read_csv(filename, sep=";", header=0, dtype={"stock_no": str})
    else:    
        df_predict = df_predict.merge(df_real, on="stock_no", how="left")

        # 实际达到的最低价
        # df_predict["low_rate"] = c_round(
        #     (df_predict["LOW_1"] - df_predict["CLOSE_price"]) / df_predict["CLOSE_price"]
        # )
        
        # 买入价格 buy_prices
        df_predict["succ_buy_cnt"] = df_predict.apply(
                lambda x: len([float(p) for p in x['buy_prices'].split(",") if float(p)>=x["LOW_1"] ]) ,
                axis=1)
        df_predict["succ_buy_price"] = df_predict.apply(
                lambda x: min([float(p) for p in x['buy_prices'].split(",") if float(p)>=x["LOW_1"]]) if x['succ_buy_cnt']>0 else 0,
                axis=1)
        
        # 按买入价格计算收益，t+1的最大收益
        df_predict[f"high2_rate"] = df_predict.apply(
            lambda x: c_round((x['high_2'] - x['succ_buy_price'])/x["CLOSE_price"]) if x['succ_buy_cnt']>0 else 0 ,
            axis=1,
        )
        # 按买入价格计算收益，t+2的最大收益
        df_predict[f"high3_rate"] = df_predict.apply(
            lambda x: c_round((x['high_3'] - x['succ_buy_price'])/x["CLOSE_price"]) if x['succ_buy_cnt']>0 else 0 ,
            axis=1,
        )
        
        # 成功卖出的价格，selL_price
        df_predict["succ_sell_cnt"] = df_predict.apply(
                lambda x: len([float(p) for p in x['sell_prices'].split(",") if float(p)<=x["HIGH_1"]]) ,
                axis=1)
        df_predict["succ_sell_price"] = df_predict.apply(
                lambda x: max([float(p) for p in x['sell_prices'].split(",") if float(p)<=x["HIGH_1"]]) if x['succ_sell_cnt']>0 else 0,
                axis=1)
        # 按昨日收盘价，卖出价的盈利
        df_predict[f"high1_rate"] = df_predict.apply(
            lambda x: c_round((x['succ_sell_price'] - x['CLOSE_price'])/x["CLOSE_price"]) if x['succ_sell_cnt']>0 else 0 ,
            axis=1,
        )
        
        df_predict.to_csv(filename, sep=";", index=False)
    
    return df_predict


def merge_predict_true(start_date=20231017):
    conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)
    sql = f"select distinct trade_date from stock_raw_daily_2 where trade_date>{start_date}"
    trade_dates = (
        pd.read_sql(sql, conn)["trade_date"].sort_values(ascending=True).tolist()
    )
    count = len(trade_dates)
    
    for idx, trade_date in enumerate(trade_dates):
        print(trade_date)
        if idx + 2 >= count:
            break

        # 第一天，T 昨收，开盘价，最低价
        df_real = pd.read_sql(
            f"select stock_no,OPEN_price as OPEN_1,LOW_price as LOW_1,HIGH_price as HIGH_1 from stock_raw_daily_2 where trade_date={trade_dates[idx]}",
            conn,
        )

        # 第二天， T+1
        df_real = df_real.merge(
            pd.read_sql(
                f"select stock_no,HIGH_price as high_2,LOW_price as low_2 from stock_raw_daily_2 where trade_date={trade_dates[idx+1]}",
                conn,), on="stock_no", how="left",
        )

        # 第三天，T+2
        df_real = df_real.merge(
            pd.read_sql(
                f"select stock_no,HIGH_price as high_3,LOW_price as low_3 from stock_raw_daily_2 where trade_date={trade_dates[idx+2]}",
                conn,
            ), on="stock_no", how="left",
        )

        # df_predict_v1 = pd.read_csv(f"data/predict/predict_merged.txt.{trade_date}",sep=";",header=0,dtype={'stock_no': str})
        # df_predict_v1 = df_predict_v1.rename(columns={'low1.7':'point_low1'})
        # merge_one(trade_date,df_predict_v1,df_real,'v1')

        df_predict_v2 = pd.read_csv(
            f"data/predict_v2/predict_merged.txt.{trade_date}", sep=";", header=0, dtype={"stock_no": str},
        )
        
        df_predict_v2 = merge_one(trade_date, df_predict_v2, df_real, "v2")
        
        sel_fields = "pk_date_stock,stock_no,succ_buy_cnt,high2_rate,high3_rate,succ_sell_cnt,high1_rate".split(",")
        order_fields = "list_dates,point,point2pair_dates,point_high1".split(",")
        for topN in [5,10]:
            for orderby in  order_fields:
                df = df_predict_v2.sort_values(by=["top3", orderby], ascending=False)[sel_fields].head(topN) #
                df['order_by'] = orderby
                df.to_csv(f"data/predict_v2/m_v2_{topN}_{orderby}_{trade_date}", sep=";", index=False, header=None)


def show_static(merge=True):
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html
    sel_fields = "pk_date_stock,stock_no,succ_buy_cnt,high2_rate,high3_rate,succ_sell_cnt,high1_rate,order_by".split(",")
    for topN in [5,10]:
        print(f'## === topN={topN} ===')
        filename = f'data/predict_v2/mm_v2_{topN}'
        if merge:
            os.system(f"cat data/predict_v2/m_v2_{topN}_* > {filename}") 
        df = pd.read_csv(filename, sep=";", header=None, names=sel_fields, dtype={"stock_no": str})
        
        # 买入统计
        for val_field in ['high2_rate','high3_rate']:
            print(f'### buy: {filename},topN={topN},val_field={val_field}')
            
            table = pd.pivot_table(df, values=[val_field], index=['order_by'], fill_value = 0,
                        columns=['succ_buy_cnt'],aggfunc="mean",sort=True, margins=True)
            print(table)
            
            table = pd.pivot_table(df, values=[val_field], index=['order_by'], fill_value = 0,
                        columns=['succ_buy_cnt'],aggfunc="count",sort=True, margins=True)
            # table['% succ_buy_cnt'] = (table['1']/table['All'])*100
            print(table)
            print("\n")
        
        # 卖出统计
        for val_field in ['high1_rate']:
            print(f'### sell: {filename},topN={topN},val_field={val_field}')
            table = pd.pivot_table(df, values=[val_field], index=['order_by'], fill_value = 0,
                        columns=['succ_sell_cnt'],aggfunc="mean",sort=True, margins=True)
            print(table)
            
            table = pd.pivot_table(df, values=[val_field], index=['order_by'], fill_value = 0,
                        columns=['succ_sell_cnt'],aggfunc="count",sort=True, margins=True)
            print(table)
            print("\n")

# python seq_model_merge.py predict_true
if __name__ == "__main__":
    op_type = sys.argv[1]
    if op_type == "intersection":
        model_intersection()
    if op_type == "predict_true":
        merge_predict_true()
    if op_type == "pivot":
        merge = True if len(sys.argv)>2 and sys.argv[2]=="True" else False
        show_static(merge)