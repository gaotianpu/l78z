#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import sqlite3  
import pandas as pd
from common import load_stocks

PROCESSES_NUM = 5

FUTURE_DAYS = 5
PAST_DAYS = 20
FEATURES_DAYS = 10

MAX_ROWS_COUNT = 850

max_high,min_high,max_low,min_low = 0.818,-0.6,0.668,-0.6

conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)

def get_max_trade_date(conn):
    trade_date = 0
    c = conn.cursor()
    cursor = c.execute("select max(trade_date) from stock_for_rnn;")
    for row in cursor:
        trade_date = row[0]
    cursor.close()
    return trade_date

max_trade_date = get_max_trade_date(conn)

def process_row(df, idx):
    if idx >= (len(df)-PAST_DAYS):
        return False

    # a.数据标识,qid=date+stockno
    trade_date = df.loc[idx]['trade_date']
    if trade_date < max_trade_date:
        return 

    # output: trade_date;stock_no;y1;y2;y3;day1_f1,day2_f2,day3_f3;day2_f1,day2_f2...;
    output_fields = []
    output_fields.append(trade_date)  # 当前日期
    output_fields.append(df.loc[idx]['stock_no'])  # stock

    # b.Y值: 要预测的未来值, 未来FUTURE_DAYS内最高价和最低价的中位数
    high_val,low_val,high_label,low_label=('nan','nan','nan','nan')
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

        high_label = round(16*(high_val-min_high)/(max_high-min_high))
        low_label = round(16*(low_val-min_high)/(max_low-min_low))

    output_fields.append(high_val)
    output_fields.append(low_val)
    output_fields.append(high_label)
    output_fields.append(low_label)

    # c. 特征值
    df_past = df.loc[idx:idx+PAST_DAYS]
    df_past_filter = df_past[df_past.HIGH > 0].describe()
    max_price = df_past_filter.loc['max']['HIGH']
    min_price = df_past_filter.loc['min']['LOW']
    max_min_price = max_price - min_price + 0.000001 #防止出现分母为0的情况

    # 过去20天内平均 ('TURNOVER', '换手率')
    max_TURNOVER = df_past_filter.loc['max']["TURNOVER"]
    min_TURNOVER = df_past_filter.loc['min']["TURNOVER"]
    max_min_TURNOVER = max_TURNOVER - min_TURNOVER + 0.000001 

    # 成交量
    max_VATURNOVER = df_past_filter.loc['max']["VATURNOVER"]
    min_VATURNOVER = df_past_filter.loc['min']["VATURNOVER"]
    max_min_VATURNOVER = max_VATURNOVER - min_VATURNOVER  + 0.000001 

    # 成交金额
    max_VOTURNOVER = df_past_filter.loc['max']["VOTURNOVER"]
    min_VOTURNOVER = df_past_filter.loc['min']["VOTURNOVER"]
    max_min_VOTURNOVER = max_VOTURNOVER - min_VOTURNOVER + 0.000001 
    
    for i in range(idx, idx+FEATURES_DAYS):
        fields = []
        fields.append( (df_past.loc[i]['TOPEN'] - min_price)/max_min_price )
        fields.append((df_past.loc[i]['TCLOSE'] - min_price)/max_min_price)
        fields.append( (df_past.loc[i]['HIGH'] - min_price)/max_min_price )
        fields.append( (df_past.loc[i]['LOW'] - min_price)/max_min_price ) 

        #当天的波动情况
        tmp_min = df_past.loc[i]['LOW']
        tmp_max_min = df_past.loc[i]['HIGH'] - df_past.loc[i]['LOW'] + 0.000001
        fields.append((df_past.loc[i]['TOPEN'] - tmp_min)/tmp_max_min)
        fields.append((df_past.loc[i]['TCLOSE'] - tmp_min)/tmp_max_min) 

        # 换手率，成交量，成交金额的情况 
        fields.append((df_past.loc[i]['TURNOVER'] - min_TURNOVER)/ max_min_TURNOVER)
        fields.append((df_past.loc[i]['VOTURNOVER'] - min_VOTURNOVER)/(max_min_VOTURNOVER))
        fields.append((df_past.loc[i]['VATURNOVER'] - min_VATURNOVER)/(max_min_VATURNOVER))

        output_fields.append(",".join([str(round(val,3)) for val in fields]))
        

    print(";".join([str(val) for val in output_fields]))
    return output_fields

def process_stock(stock_no, data_type="train"):
    max_count = MAX_ROWS_COUNT + PAST_DAYS + 10 if data_type=="train" else PAST_DAYS + 10
    df = pd.read_sql("select * from stock_raw_daily where stock_no='%s' order by trade_date desc limit 0,%d;"%(stock_no,max_count), conn)
    
    end_idx = len(df)-PAST_DAYS
    if data_type == "predict":
        process_row(df, 0)
    else:  # train
        i = 0
        for idx in range(FUTURE_DAYS, end_idx):
            try:
                trade_date = df.loc[idx]['trade_date']
                if trade_date < max_trade_date:
                    break

                process_row(df, idx)
                i = i + 1
                if i > MAX_ROWS_COUNT:
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

if __name__ == "__main__":
    data_type = sys.argv[1]
    process_idx = -1 if len(sys.argv) != 3 else int(sys.argv[2])
    process_all_stocks(data_type, process_idx)
    # process_stock('300782')
    conn.close()
