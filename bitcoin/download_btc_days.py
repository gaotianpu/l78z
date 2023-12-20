#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
历史数据下载,临时的
"""
import os
import sys
import time
import random
import json
import re
import requests
import pandas as pd 
import datetime
# from datetime import datetime, timezone, timedelta
import logging
from multiprocessing import Pool
from itertools import islice
import sqlite3  

def download(year=2023):
    cache_file = f"data/btc/{year}.html"
    if os.path.exists(cache_file):
        with open(cache_file,'r') as f:
            return f.read()

    source_url = f"https://history.btc123.fans/data/btc/{year}/"
    resp = requests.get(url=source_url, timeout=3.5)
    with open(cache_file,'w') as f:
        f.write(resp.text)
        
    return resp.text


p_td = re.compile(r">([^<^']+)")
def parse(year):
    txt = download(year)
    
    start_str = '<tr align=center><td><a target=_blank'
    start_idx = txt.find(start_str)
    end_str = '<div class="layui-tab-item">'
    end_idx = txt.find(end_str)
    tr_rows = txt[start_idx:end_idx].split("<tr align=center>")
    
    all_rows = []
    for row in tr_rows:
        td_fields =  row.split("</td>")
        if len(td_fields)!=8:
            continue
            
        fields = [p_td.search(val).groups(0)[0] for val in td_fields[:-1]]
        fields[0] = fields[0].replace("-","")
        fields[5] = float(fields[5].replace(",",""))/10000
        fields[6] = fields[6].replace("%","")
        
        fields = [float(f) for f in fields]
        fields[0] = int(fields[0])
        # print(fields)
        fields.insert(1,1) #btc_no=1
        all_rows.append(fields)
    return all_rows

def process():
    all_rows = []
    for year in range(2014,2024):
        all_rows = all_rows + parse(year)
        
    columns="trade_date,btc_no,open,high,low,close,amount,rate".split(",")
    df = pd.DataFrame(all_rows,columns=columns)
    df = df.sort_values(by=["trade_date"],ascending=False)
    df = df.reset_index(drop=True) #排序后重置索引
    
    df = compute_more_rate(df)
    
    df.to_csv(f"data/btc/all_2014_2023.csv",sep=";",index=None) #header=None,

price_fields = "open,low,high,close".split(",")

def compute_more_rate(df):
    '''持有天数与收益情况
    '''
    for p1 in price_fields:
        for p2 in price_fields:
            df[f'delta_{p1}_{p2}']=0.0
    df['delta_amount']=0.0
    df['range_price']=0.0
    
    for i in range(1,7):     
        df[f'rate_{i}']=0.0
    
    total_cnt = len(df)     
    for idx,row in df.iterrows():
        if idx+1<total_cnt:
            for p1 in price_fields:
                for p2 in price_fields:
                    df.loc[idx,f'delta_{p1}_{p2}'] = round((row[p1] - df.loc[idx+1][p2])*100/df.loc[idx+1][p2],4)
            df['delta_amount'] = round((row["amount"] - df.loc[idx+1]["amount"])*100/df.loc[idx+1]["amount"],4)
            df.loc[idx,f'range_price'] = round((row["high"] - row["low"])*100/df.loc[idx+1]["close"],4) #波动幅度
            
            # 比特币24时小时连续交易，last_close==open, 下面股票里的似乎没必要？
            # df.loc[idx,f'high_topen'] = round((row["high"] - row["open"])*100/row["open"],4) #波动幅度
            # df.loc[idx,f'low_topen'] = round((row["low"] - row["open"])*100/row["open"],4) #波动幅度
            # df.loc[idx,f'close_topen'] = round((row["close"] - row["open"])*100/row["open"],4) #波动幅度
            
           
        for days in range(6):
            day_span = days + 1
            new_idx = idx + day_span
            if (new_idx+1)>=total_cnt:
                break 
            rate = round((row['close'] - df.loc[new_idx]['open'])*100/df.loc[new_idx]['open'],4)
            # 当日开盘价(和rate保持一致) new_idx？or 第二日开盘价(预测用) new_idx+1？
            df.loc[new_idx+1,f'rate_{day_span}'] = rate
            
            # print(df.loc[new_idx]['trade_date'],df.loc[new_idx]['open'],row['trade_date'],row['close'],rate)
    return df 

def statics():
    df = pd.read_csv(f"data/btc/all_2014_2023.csv",sep=";",header=0,dtype={'trade_date':int})
    
    d = df.describe()
    print(d)
    print(d['high']['max'],d['low']['min'])
    
    # df.to_csv(f"data/btc/all_more_rate_2014_2023.csv",sep=";",index=None) #header=None,
    
    # 2. 每年的均值，标准差，4分位情况
    # 3. 比特币价格与美元指数的关系
    # pass

if __name__ == "__main__":
    process()
    # statics()
    