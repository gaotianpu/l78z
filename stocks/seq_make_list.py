import os
import sys
import json
import random
import pandas as pd
import sqlite3

from common import load_stocks,load_trade_dates

conn = sqlite3.connect("file:data/stocks_train_2.db?mode=ro", uri=True) 

# (6959210-1960*2*24)/1960/64 = 54.7
SAMPLE_COUNT_PER_DAY = 66
GROUP_ITEMS_COUNT = 24

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

def load_ids_by_date(date,dateset_type=0):
    sql = "select pk_date_stock,list_label from stock_for_transfomer where trade_date='%s' and dataset_type='%d'" %(date,dateset_type)
    df = pd.read_sql(sql, conn)
    return df

def load_ids_by_stock(stock_no,dateset_type=0):
    sql = "select pk_date_stock,list_label from stock_for_transfomer where stock_no='%s' and dataset_type='%d'" %(stock_no,dateset_type)
    df = pd.read_sql(sql, conn)
    return df

def via_3_8(label_dfs):
    cnt = 3 #3*8=24，每个档位抽3个，共计抽24个, 相当提升了高档位的占比，让模型更多关注高档位部分
    selected_ids = []
    for label,df in enumerate(label_dfs): 
        # df = label_dfs[label] #date_df[date_df['list_label']==(label+1)] #load_ids_by_date_label(date,0,label+1)
        if len(df) > cnt*2:
            c_ids = df["pk_date_stock"].sample(n=cnt).values.tolist()
            selected_ids = selected_ids + c_ids
        # else:
        #     print("error validate data cnt less date=%s label=%s"%(date,label))
    if len(selected_ids)==GROUP_ITEMS_COUNT:
        print(",".join( [str(v) for v in selected_ids] ))

def via_22223355(label_dfs):
    # cnts = [5,5,3,3,2,2,2,2] #注意顺序
    cnts = [3,3,2,2,2,2,2,2,2,1,1,1,1] #总计24,
    selected_ids = []
    for label,df in enumerate(label_dfs): #range(8): 
        cnt = cnts[label]
        if len(df) > cnt:
            c_ids = df["pk_date_stock"].sample(n=cnt).values.tolist()
            selected_ids = selected_ids + c_ids
    if len(selected_ids)==24:
        print(",".join( [str(v) for v in selected_ids] ))
    
def gen_date_list(cnt_type='22223355'):
    trade_dates = load_trade_dates(conn,0) # start_date=0
    for idx,date in enumerate(trade_dates): 
        df = load_ids_by_date(date,0)
        
        label_dfs = []
        for label in range(13):
            label_dfs.append(df[df['list_label']==label]) #load_ids_by_date_label(date,0,label+1)
        
        for i in range(SAMPLE_COUNT_PER_DAY*3):
            if cnt_type=='3_8': 
                via_3_8(label_dfs)
            if cnt_type=='22223355': 
                via_22223355(label_dfs)
            # break
        # break

def gen_stock_list(cnt_type='22223355'):
    conn1 = sqlite3.connect("file:data/stocks.db", uri=True)
    stocks = load_stocks(conn1) 
    for idx,stock in enumerate(stocks): 
        df = load_ids_by_stock(stock[0],0)
        
        label_dfs = []
        for label in range(8): 
            label_dfs.append(df[df['list_label']==(label+1)]) #load_ids_by_date_label(date,0,label+1)
        
        for i in range(SAMPLE_COUNT_PER_DAY*3):
            if cnt_type=='3_8': 
                via_3_8(label_dfs)
            if cnt_type=='22223355': 
                via_22223355(label_dfs)
            # break
        # break

if __name__ == "__main__":
    op_type = sys.argv[1]
    # print(op_type)
    if op_type == "random":
        # python seq_make_list.py random > list/train_radom.txt
        gen_via_random()
    if op_type == "date":
        # python seq_make_list.py date > data2/list.date.train_22223355.txt
        gen_date_list("22223355")
    if op_type == "stocks":
        # python seq_make_list.py stocks > data2/list.stocks.train_22223355.txt
        gen_stock_list("22223355")
    if op_type == "3_8":
        # 3*8形式的采样和实际的数据分布相差较远，不再采用
        # python seq_make_list.py 3_8 > list/train_3_8.txt
        gen_date_list(op_type)
    
        