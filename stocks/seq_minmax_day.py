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

# day_minmax/stocks.db
conn = sqlite3.connect("file:newdb/stocks.db?mode=ro", uri=True)

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
    
    tmp = df_past.sort_values(by=['HIGH_price'],ascending=False).head(1)  #.iloc[0] 
    idx = tmp.index.tolist()[0]
    next_idx = idx-1 if idx>0 else idx
    maxli = [stock_no,tmp.iloc[0]['trade_date'],df.loc[next_idx]['trade_date'],1,tmp.iloc[0]['HIGH_price']]
    
    tmp = df_past.sort_values(by=['LOW_price'],ascending=True).head(1)
    idx = tmp.index.tolist()[0]
    next_idx = idx-1 if idx>0 else idx 
    minli = [stock_no,tmp.iloc[0]['trade_date'],df.loc[next_idx]['trade_date'],0,tmp.iloc[0]['LOW_price']]
    return maxli,minli

def process_stock(stock_no):
    fname = f"day_minmax/{stock_no}.txt"
    if os.path.exists(fname):
        # df = pd.read_csv(fname,sep=";",header=0)
        # df = df.drop_duplicates()
        # df.to_csv(f"day_minmax_uniq/{stock_no}.txt",sep=";",index=False,header=None)
        pass
        # continue
            
    sql = f"select * from stock_raw_daily where stock_no='{stock_no}' and OPEN_price>0 order by trade_date desc"
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
        # if page_idx==1:
        # break
            
    vdf = pd.DataFrame(stock_li,columns=['stock_no','trade_date',"next_date",'Min_max','price'])
    vdf = vdf.sort_values(by=['trade_date'],ascending=False)
    vdf = vdf.drop_duplicates()
    vdf = vdf.reset_index(drop=True) #排序后重置索引
    
    vdf['is_deleted'] = 0 #删除重复的: 相邻都是1，取最高值那行，其他标记为删除； 都是0，取最低值，其他标记删除
    
    last_indexs = []
    last_prices = []
    last_minmax = -1
    for index, row in vdf.iterrows():
        if row['Min_max']!=last_minmax:
            if len(last_indexs)>1:
                vdf.loc[last_indexs, "is_deleted"] = 1
                save_idx = -1
                if last_minmax==0: #最小值
                    save_idx = last_indexs[last_prices.index(min(last_prices))] 
                if last_minmax==1: #最大值
                    save_idx = last_indexs[last_prices.index(max(last_prices))] 
                if save_idx>0:
                    vdf.loc[save_idx, "is_deleted"] = 0 
                    
            last_indexs = [index]
            last_prices = [row['price']]
            last_minmax = row['Min_max']
            continue
        
        last_indexs.append(index)
        last_prices.append(row['price'])
        
        
    vdf = vdf[vdf['is_deleted'] == 0]
    vdf = vdf.drop(['is_deleted'],axis=1)
    vdf = vdf.reset_index(drop=True)
    
    vdf['rate'] = 0 
    last_row = None 
    for index, row in vdf.iterrows():
        if index>0:
            vdf.loc[index, "rate"] = round((last_row['price'] - row['price'])/row['price'],4)
        last_row = row 
    
    vdf.to_csv(fname,sep=";",index=False,header=None)
    
def process():
    stocks = load_stocks(conn)
    for i, stock in enumerate(stocks):
        stock_no = stock[0]
        process_stock(stock_no) 
        
        # break

if __name__ == "__main__":
    # data_type = sys.argv[1]
    # if data_type == "train": 
    process()