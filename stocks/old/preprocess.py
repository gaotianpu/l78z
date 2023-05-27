#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理
作为回归问题，分别预测未来5天内的最高价和最低价

1. 训练集、验证集、测试集的划分？
最近第二天的作为验证集
最近第一天的作为测试集
其他历史数据统一作为训练集

2. 特征选取

3. 最终字段
stock_no,date,future_high_price,future_low_price,


"""
import os
import sys
from datetime import datetime
import pandas as pd
from common import load_stocks


PROCESSES_NUM = 5

FUTURE_DAYS = 5
PAST_DAYS = 20
FEATURES_DAYS = 5

CN_NAMES = "日期,股票代码,收盘价,最高价,最低价,开盘价,前收盘,涨跌额,涨跌幅,换手率,成交量,成交金额,总市值,流通市值".split(
    ",")
NAMES = "DATE;STOCK;TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP".split(
    ";")

HIGH_LABEL_VAL_RANGES = [0.067,0.041,0.027,0.016,0.007,-0.003,-0.016]
LOW_LABEL_VAL_RANGES = [0.012, 0.0, -0.015, -0.034]


def convert_level(val, val_ranges):
    """连续值分桶成离散值"""
    level = 0
    for i, label in enumerate(val_ranges):
        if val > label:
            level = len(val_ranges) - i
            break
    return level


def process_row(df, idx):
    if idx >= (len(df)-PAST_DAYS):
        return False

    # a.数据标识,qid=date+stockno
    output_fields = []
    output_fields.append(df.loc[idx]['DATE'])  # 当前日期
    output_fields.append(df.loc[idx]['STOCK'])  # stock

    # b.Y值: 要预测的未来值, 未来FUTURE_DAYS内最高价和最低价的中位数
    high_val,low_val=('nan','nan')
    if idx >= 2:
        topen = df.loc[idx-1]['TOPEN']
        if topen == 0:
            return False

        tclose = df.loc[idx-1]['TOPEN']

        df_future = df.loc[idx-FUTURE_DAYS:idx-1]
        # print(df_future)
        desc50 = df_future[df_future.HIGH > 0].describe().loc['50%']
        high_val = round((desc50['HIGH']-topen)/topen, 3) 
        low_val = round((desc50['LOW']-topen)/topen, 3) 

    output_fields.append(high_val)
    output_fields.append(low_val)

    # c. 特征值
    df_past = df.loc[idx:idx+PAST_DAYS]
    df_past_filter = df_past[df_past.HIGH > 0].describe()
    max_price = df_past_filter.loc['max']['HIGH']
    min_price = df_past_filter.loc['min']['LOW']
    max_min_price = round(max_price - min_price,3)
    # print(max_price,min_price,max_min_price)
    # output_fields.append(max_price)  # 
    # output_fields.append(min_price)  # 

    # 过去20天内平均 ('TURNOVER', '换手率')
    max_TURNOVER = round(df_past_filter.loc['max']["TURNOVER"],3)
    min_TURNOVER = round(df_past_filter.loc['min']["TURNOVER"],3)
    # 成交量
    max_VATURNOVER = round(df_past_filter.loc['max']["VATURNOVER"],3)
    min_VATURNOVER = round(df_past_filter.loc['min']["VATURNOVER"],3)
    # 成交金额
    max_VOTURNOVER = round(df_past_filter.loc['max']["VOTURNOVER"],3)
    min_VOTURNOVER = round(df_past_filter.loc['min']["VOTURNOVER"],3)

    # output_fields.append(max_TURNOVER)  # 
    
    for i in range(idx, idx+FEATURES_DAYS):  # f12 ~ 15  price
        # print(i,df_past.loc[i]['DATE'])
        # TCLOSE;HIGH;LOW;TOPEN
        if max_min_price > 0:
            output_fields.append(
                round((df_past.loc[i]['TOPEN'] - min_price)/max_min_price, 3))
            output_fields.append(
                round((df_past.loc[i]['TCLOSE'] - min_price)/max_min_price, 3))
            output_fields.append(
                round((df_past.loc[i]['HIGH'] - min_price)/max_min_price, 3))
            output_fields.append(
                round((df_past.loc[i]['LOW'] - min_price)/max_min_price, 3))
        else :
            output_fields.append(-1)
            output_fields.append(-1)
            output_fields.append(-1)
            output_fields.append(-1)


        #当天的波动情况
        tmp_min = df_past.loc[i]['LOW']
        tmp_max_min = df_past.loc[i]['HIGH'] - df_past.loc[i]['LOW']
        if tmp_max_min>0:
            output_fields.append(
                round((df_past.loc[i]['TOPEN'] - tmp_min)/tmp_max_min, 3))
            output_fields.append(
                round((df_past.loc[i]['TCLOSE'] - tmp_min)/tmp_max_min, 3))
        else:
            output_fields.append(-1)
            output_fields.append(-1)     

        # 换手率，成交量，成交金额的情况
        if (max_TURNOVER-min_TURNOVER)>0:
            output_fields.append(
                round((df_past.loc[i]['TURNOVER'] - min_TURNOVER)/(max_TURNOVER-min_TURNOVER), 3))
            output_fields.append(
                round((df_past.loc[i]['VOTURNOVER'] - min_VOTURNOVER)/(max_VOTURNOVER-min_VOTURNOVER), 3))
            output_fields.append(
                round((df_past.loc[i]['VATURNOVER'] - min_VATURNOVER)/(max_VATURNOVER-min_VATURNOVER), 3))
        else:
            output_fields.append(-1)
            output_fields.append(-1)
            output_fields.append(-1)

    print(",".join([str(val) for val in output_fields]))
    return output_fields


def process_stock(stock_no, data_type="train"):
    data_file = "data/history/%s.csv" % (stock_no)
    if not os.path.isfile(data_file):
        return

    df = pd.read_csv(data_file,
                     names=NAMES, dtype={'STOCK': str})

    end_idx = len(df)-PAST_DAYS
    if data_type == "predict":
        process_row(df, 0)
    elif data_type == "test":
        process_row(df, FUTURE_DAYS)
    # elif data_type == "validate":
    #     process_row(df, FUTURE_DAYS+1)
    else:  # train
        i = 0
        for idx in range(FUTURE_DAYS+1, end_idx):
            try:
                process_row(df, idx)
                i = i + 1
                if i > 800:
                    break 
            except:
                print("except:%s_%d" % (stock_no, idx), file=sys.stderr)
                continue
    df = None  # 释放内存？
    del df


def process_all_stocks(data_type="train", processes_idx=-1):
    stocks = load_stocks()
    for i, stock in enumerate(stocks):
        if processes_idx < 0:
            process_stock(stock[0], data_type)
        elif i % PROCESSES_NUM == processes_idx:
            process_stock(stock[0], data_type)
        # if i > 100:
        #     break 


def test():
    df = pd.read_csv("data/history/300782.csv",
                     names=NAMES, dtype={'STOCK': str})
    idx = FUTURE_DAYS #2
    # process_row(df, idx)

    # process_stock("300782","predict")
    # process_stock("300782","test")
    process_stock("300782","train")


# run_train.sh
if __name__ == "__main__":
    data_type = sys.argv[1]
    process_idx = -1 if len(sys.argv) != 3 else int(sys.argv[2])
    process_all_stocks(data_type, process_idx)
    # test()
    
