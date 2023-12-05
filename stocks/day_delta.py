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

past_days = 15 
future_days = 5 

def minmax(x,min,max):
    return round((x-min)/(max-min+0.00000001),4)

def zscore(x,mean,std):
    return round((x-mean)/(std+0.00000001),4)

def process_row(df,idx):
    trade_date = df["trade_date"][idx]
    stock_no = df['stock_no'][idx]
    pk_date_stock = int(str(trade_date) + stock_no)
    
    fields = [pk_date_stock,trade_date,stock_no]
    
    # 预测数据
    high_rate = 0.0 
    if idx>=future_days:
        high_rate = c_round((df['HIGH_price'][idx-future_days] - df['HIGH_price'][idx])/df['HIGH_price'][idx])
    fields.append(high_rate)
    
    # 历史数据，输入向量   
    df_past = df.loc[idx:idx+past_days].copy()
    
    # 1. 归一化处理
    (price_min,price_max) = (df_past['LOW_price'].min(),df_past['HIGH_price'].max())
    if price_min == price_max:
        print(f"price_min == price_max:{stock_no} {pk_date_stock}")
        return False
        
    for field in "OPEN_price,CLOSE_price,LOW_price,HIGH_price".split(","):
        df_past[f"{field}_minmax"] = df_past.apply(lambda x: minmax(x[field],price_min,price_max) , axis=1)
    
    field_minmax={}
    for field in "TURNOVER,TURNOVER_amount,TURNOVER_rate".split(","):
        field_minmax[field] = (df_past[field].min(),df_past[field].max())
        if field_minmax[field][0] == field_minmax[field][1]:
            print(f" {field} min == max:{stock_no} {pk_date_stock}")
            # return False
            
        df_past[f"{field}_minmax"] = df_past.apply(lambda x: minmax(x[field],field_minmax[field][0],field_minmax[field][1]) , axis=1)
    
    # 2. 计算mean,std, 作为向量字段，同时用于最后一天数据计算z-score
    all_fields = []
    field_meanstd={}
    for field in "OPEN_price,CLOSE_price,LOW_price,HIGH_price,TURNOVER,TURNOVER_amount,TURNOVER_rate".split(","):
        field_meanstd[field] = (df_past[f"{field}_minmax"].mean(),df_past[f"{field}_minmax"].std())
        all_fields.append(c_round(df_past[f"{field}_minmax"].mean()))
        all_fields.append(c_round(df_past[f"{field}_minmax"].std()))
    
    # 3. 计算最后一个交易日的各种字段的z-score
    last_day = df_past.loc[idx]
    for field in "OPEN_price,CLOSE_price,LOW_price,HIGH_price,TURNOVER,TURNOVER_amount,TURNOVER_rate".split(","):
        score = zscore(last_day[f'{field}_minmax'],field_meanstd[field][0],field_meanstd[field][1])
        all_fields.append(score)
    
    # 4. 最后一个交易日的各种rate,
    last2_day = df_past.loc[idx+1]
    all_fields.append(c_round( (last_day['TURNOVER']-last2_day['TURNOVER'])/last2_day['TURNOVER'] ))
    all_fields.append(c_round( (last_day['TURNOVER_amount']-last2_day['TURNOVER_amount'])/last2_day['TURNOVER_amount'] ))
    all_fields.append(c_round( (last_day['TURNOVER_rate']-last2_day['TURNOVER_rate'])/(last2_day['TURNOVER_rate']+0.00000001) ))
    all_fields.append(c_round( (last_day['OPEN_price']-last2_day['CLOSE_price'])/last2_day['CLOSE_price'] ))
    all_fields.append( c_round( (last_day['HIGH_price']-last2_day['HIGH_price'])/last2_day['HIGH_price'] ))
    all_fields.append( c_round( (last_day['LOW_price']-last2_day['LOW_price'])/last2_day['LOW_price'] ))
    all_fields.append(c_round( (last_day['LOW_price']-last_day['OPEN_price'])/last_day['OPEN_price']  ))
    all_fields.append(c_round( (last_day['HIGH_price']-last_day['OPEN_price'])/last_day['OPEN_price']  ))
    all_fields.append(c_round( (last_day['CLOSE_price']-last_day['OPEN_price'])/last_day['OPEN_price']  ))
    
    
    if len(all_fields)==30:  
        fields.append(",".join([str(field) for field in all_fields]))
        return fields
    else:
        print("len(all_fields)!=21")
        return False

def process():
    stocks = load_stocks(conn)
    
    for i, stock in enumerate(stocks):
        stock_no = stock[0]
        fname = f"day_delta/{stock_no}.txt"
        if os.path.exists(fname):
            continue
        print(stock_no)
        sql = f"select * from stock_raw_daily_2 where stock_no='{stock_no}' and OPEN_price>0 order by trade_date desc"
        df = pd.read_sql(sql, conn)
        end_idx = len(df) - past_days + 1
        
        stock_li = []
        for idx in range(future_days, end_idx):
            ret = process_row(df, idx)
            if ret:
                stock_li.append(ret)
             
        vdf = pd.DataFrame(stock_li)
        vdf = vdf.sort_values(by=[3],ascending=False)
        vdf.to_csv(fname,sep=";",index=False,header=None)
        # break 

def process_predict():
    stocks = load_stocks(conn)
    
    current_date = 0 
    stock_li = []
    for i, stock in enumerate(stocks):
        stock_no = stock[0]
        cnt = past_days + future_days
        sql = f"select * from stock_raw_daily_2 where stock_no='{stock_no}' and OPEN_price>0 order by trade_date desc limit 0,{cnt}"
        df = pd.read_sql(sql, conn) 
        current_date = int(df.loc[0]['trade_date'])
        ret = process_row(df, 0)
        if ret:
            stock_li.append(ret)
            
    vdf = pd.DataFrame(stock_li)
    vdf = vdf.sort_values(by=[3],ascending=False)
    vdf.to_csv(f"day_delta.{current_date}.txt",sep=";",index=False,header=None) 
        
        # 
        
        
        # fname = f"day_delta/{stock_no}.txt"
        # if os.path.exists(fname):
        #     continue
        
        # print(stock_no)
        # sql = f"select * from stock_raw_daily_2 where stock_no='{stock_no}' and OPEN_price>0 order by trade_date desc"
        # df = pd.read_sql(sql, conn)
        # end_idx = len(df) - past_days + 1
        
        # stock_li = []
        # for idx in range(future_days, end_idx):
        #     ret = process_row(df, idx)
        #     if ret:
        #         stock_li.append(ret)
             
        

def cosine():
    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity([[1, 0, 0, 0],[2,1,1,1]], [[1, 0, 0, 0]])  
    # 值越大，越相似     

if __name__ == "__main__":
    process()
    # process_predict()