'''滑动计算N日内的最高价、最低价，用于寻找一只股票的关键帧
'''
import os
import sys
import time
import random
import json
# import datetime
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import requests
import re 
import logging
from multiprocessing import Pool
from itertools import islice
import sqlite3  
import pickle
from threading import Thread
from common import load_stocks,c_round

conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)

slide_windows = 10
past_days = 15 
future_days = 5 

def minmax(x,min,max):
    return round((x-min)/(max-min+0.00000001),4)

def zscore(x,mean,std):
    return round((x-mean)/(std+0.00000001),4)

def process_row(df,start_idx,end_idx):
    stock_no = df['stock_no'][0]
    df_past = df.loc[start_idx:end_idx].copy()
    
    tmp = df_past.sort_values(by=['HIGH_price'],ascending=False).head(1)
    # max_price_date = tmp['trade_date'].tolist()[0]
    idx = tmp.index.tolist()[0]
    next_idx = idx-1 if idx>0 else idx 
    # print(max_price_date,df.loc[idx-1]['trade_date'] )
    maxli = [stock_no,tmp['trade_date'].tolist()[0],df.loc[next_idx]['trade_date'],1]
    
    tmp = df_past.sort_values(by=['LOW_price'],ascending=True).head(1)
    idx = tmp.index.tolist()[0]
    next_idx = idx-1 if idx>0 else idx 
    minli = [stock_no,tmp['trade_date'].tolist()[0],df.loc[next_idx]['trade_date'],0]
    return maxli,minli

def process():
    stocks = load_stocks(conn)
    for i, stock in enumerate(stocks):
        stock_no = stock[0]
        fname = f"day_minmax/{stock_no}.txt"
        if os.path.exists(fname):
            # df = pd.read_csv(fname,sep=";",header=0)
            # df = df.drop_duplicates()
            # df.to_csv(f"day_minmax_uniq/{stock_no}.txt",sep=";",index=False,header=None)
            pass
            # continue
        
        sql = f"select * from stock_raw_daily_2 where stock_no='{stock_no}' and OPEN_price>0 order by trade_date desc"
        df = pd.read_sql(sql, conn)
        total_cnt = len(df)
        page_cnt =  int(total_cnt/slide_windows)
        page_cnt = page_cnt if total_cnt%slide_windows==0 else page_cnt+1
        
        print(stock_no,total_cnt,page_cnt)
        
        stock_li = []
        for page_idx in range(page_cnt):
            start_idx = page_idx*slide_windows
            end_idx = page_idx*slide_windows + past_days
            end_idx =  end_idx if end_idx<total_cnt else total_cnt
            # print(start_idx,end_idx)
            maxli,minli = process_row(df, start_idx,end_idx)
            stock_li.append(maxli)
            stock_li.append(minli)
            
            # if ret:
            #     stock_li = stock_li + ret
            
            # if page_idx==1:
            # break
             
        vdf = pd.DataFrame(stock_li,columns=['stock_no','trade_date',"next_date",'Min_max'])
        vdf = vdf.sort_values(by=['trade_date'],ascending=False)
        vdf = vdf.drop_duplicates()
        vdf.to_csv(fname,sep=";",index=False,header=None)
        # break

if __name__ == "__main__":
    # data_type = sys.argv[1]
    # if data_type == "train": 
    process()