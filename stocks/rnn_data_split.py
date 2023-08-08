import os
import sys
import json
import random
import pandas as pd
import sqlite3

SAMPLE_COUNT_PER_DAY = 20  #每个交易日抽取多少条作为样本

conn = sqlite3.connect("file:data/stocks.db", uri=True)

def load_trade_dates():
    sql = "select distinct trade_date from stock_for_transfomer"
    df = pd.read_sql(sql, conn)
    return df['trade_date'].tolist()

def load_ids_by_date(date,dateset_type=0):
    sql = "select pk_date_stock from stock_for_transfomer where trade_date='%s' and dataset_type='%d'" %(date,dateset_type)
    df = pd.read_sql(sql, conn)
    return df['pk_date_stock']

def upate_dataset_type(selected_ids,dataset_type):
    str_selected_ids = ",".join([str(x) for x in selected_ids])
    sql = "update stock_for_transfomer set dataset_type=%d where pk_date_stock in (%s)" %(dataset_type,str_selected_ids)
    c = conn.cursor()
    cursor = c.execute(sql)
    conn.commit()
    cursor.close()

def dataset_split(dataset_type): # 1=验证集 2=测试集
    trade_dates = load_trade_dates()
    for date in trade_dates: 
        df = load_ids_by_date(date,0) #取dataset_type=0(默认值)的数据
        selected_ids = df.sample(n=SAMPLE_COUNT_PER_DAY) #每日抽取20条 
        upate_dataset_type(selected_ids,dataset_type)
         

if __name__ == "__main__":
    dataset_split(1)
    dataset_split(2)