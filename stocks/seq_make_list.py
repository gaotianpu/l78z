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

def gen_via_random(dataset_type="train"):
    trade_dates = load_trade_dates(conn,0) # start_date=0
    for idx,date in enumerate(trade_dates): 
        for i in range(SAMPLE_COUNT_PER_DAY):
            load_pkid_by_date(date)
            # break

def load_ids_by_date_label(date,dateset_type=0,label=1,conn=conn):
    sql = "select pk_date_stock from stock_for_transfomer where trade_date=%s and dataset_type=%d and list_label=%s " %(date,dateset_type,label)
    df = pd.read_sql(sql, conn)
    return df['pk_date_stock']

def date_via_3_8(date):
    cnt = 3 #3*8=24，每个档位抽3个，共计抽24个, 相当提升了高档位的占比，让模型更多关注高档位部分
    selected_ids = []
    for label in range(8): 
        df = load_ids_by_date_label(date,0,label+1)
        # cnt = label_cnt_map[label]
        if len(df) > cnt*2:
            c_ids = df.sample(n=cnt).values.tolist()
            selected_ids = selected_ids + c_ids
        # else:
        #     print("error validate data cnt less date=%s label=%s"%(date,label))
    print(",".join( [str(v) for v in selected_ids] ))
    
def gen_via_3_8():
    trade_dates = load_trade_dates(conn,0) # start_date=0
    for idx,date in enumerate(trade_dates): 
        for i in range(SAMPLE_COUNT_PER_DAY*3):
            date_via_3_8(date)

if __name__ == "__main__":
    op_type = sys.argv[1]
    # print(op_type)
    if op_type == "random":
        # python seq_make_list.py random > list/train_radom.txt
        gen_via_random()
    if op_type == "3_8":
        # python seq_make_list.py 3_8 > list/train_3_8.txt
        gen_via_3_8()
        