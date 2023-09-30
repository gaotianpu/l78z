import os
import sys
import json
import random
import pandas as pd
import sqlite3

from common import load_stocks,load_trade_dates

conn = sqlite3.connect("file:data/stocks_train.db?mode=ro", uri=True) 

# (6959210-1960*2*24)/1960/64 = 54.7
SAMPLE_COUNT_PER_DAY = 66 

def load_pkid_by_date(date,dateset_type=0,field="f_high_mean_rate"):
    sql = "select cast(pk_date_stock as text) as pk_date_stock from stock_for_transfomer where trade_date='%s' and dataset_type='%d' order by random() limit 64" %(date,dateset_type)
    df = pd.read_sql(sql, conn)
    print( ",".join(df["pk_date_stock"].values.tolist()))

def process(dataset_type="train"):
    trade_dates = load_trade_dates(conn,0) # start_date=0
    for idx,date in enumerate(trade_dates): 
        for i in range(SAMPLE_COUNT_PER_DAY):
            load_pkid_by_date(date)
            # break

if __name__ == "__main__":
    process()
        