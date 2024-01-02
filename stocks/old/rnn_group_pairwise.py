#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import random
import sqlite3
import pandas as pd 

# train.data, group.data

# label,idx_1,idx_2
# 注意label的样本均衡问题

conn = sqlite3.connect("file:newdb/stocks.db?mode=ro", uri=True)

def get_max_trade_date(conn):
    trade_date = 0
    c = conn.cursor()
    cursor = c.execute("select max(trade_date) from stock_for_rnn_pair;")
    for row in cursor:
        trade_date = row[0]
    cursor.close()
    return trade_date

max_trade_date = get_max_trade_date(conn)


def process_oneday(df,date,data_type):
    date_rows = df[df["date"]==date] 
    li = []
    for idx_0,row_0 in date_rows.iterrows():
        for idx_1,row_1 in date_rows.iterrows():
            if idx_1 > idx_0 and abs(row_0['high']-row_1['high'])>0.15:
                print(date,row_0['stock'],row_1['stock'],data_type, 1 if row_0['high']>row_1['high'] else 0)  
                print(date,row_1['stock'],row_0['stock'],data_type, 1 if row_1['high']>row_0['high'] else 0)
    return li 


def process(data_type, date_count=200):
    df = pd.read_csv("data/rnn_%s.txt"%(data_type),
        names="date,stock,high,low,high_label,low_label,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16".split(","),
        header=None, dtype={'stock':str,'high_label':int}, sep=";")

    dates = df["date"].unique()

    dt = 0 if data_type=="train" else 1
    for i,date in enumerate(dates):
        if date < max_trade_date:
            break
        process_oneday(df,date,dt)
        if i>date_count:
            break

if __name__ == "__main__":
    data_type = sys.argv[1]
    date_count = int(sys.argv[2]) 
    process(data_type,date_count)