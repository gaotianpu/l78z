#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

FUTURE_DAYS = 5 # 预测未来几天的数据, 2,3,5? 2比较合适，3则可能出现重复，在不恰当的数据集划分策略下，训练集和测试可能会重叠？
PAST_DAYS = 20 #使用过去几天的数据做特征
MINMAX_DAYS = 60

FIELDS_PRICE="open;high;low;close".split(";")
FIELDS="amount;rate;delta_open_open;delta_open_low;delta_open_high;delta_open_close;delta_low_open;delta_low_low;delta_low_high;delta_low_close;delta_high_open;delta_high_low;delta_high_high;delta_high_close;delta_close_open;delta_close_low;delta_close_high;delta_close_close;delta_amount;range_price".split(";")

def minmax(x,min,max):
    if max == min:
        return 0.00000001
    val = round((x-min)/(max-min),4)
    if val>1:
        val = 1
    if val<0:
        val = 0
    return val

def zscore(x,mean,std):
    return round((x-mean)/(std+0.00000001),4)

def compute_rate(x,base): #计算涨跌幅度
    return round((x-base)*100/base,4)

def process_row(df, idx,current_date,need_print=True):
    ret = {"btc_no": "1", "current_date":current_date}
    (highN_rate,high1_rate,low1_rate) = (0.0,0.0,0.0)
    
    # 未来值,FUTURE_DAYS最高价，最低价？
    if idx>0: #train
        buy_base = df.loc[idx-1]['open'] # df_future.iloc[-1]['TOPEN']
        hold_base = df.loc[idx]['close'] #昨收价格
        
        #保守点，把最高价?作为买入价，用future_days的最低价减去当前的最高价
        highN_rate = compute_rate(df.loc[idx-FUTURE_DAYS]['high'],buy_base)     
        lowN_rate = compute_rate(df.loc[idx-FUTURE_DAYS]['low'],buy_base)     
              
        #用于预测要买入的价位
        low1_rate = compute_rate(df.loc[idx-1]['low'],hold_base)
        #用于预测要卖出的价位
        high1_rate = compute_rate(df.loc[idx-1]['high'],hold_base)

        ret['highN_rate'] = highN_rate 
        ret['lowN_rate'] = lowN_rate  #做空机制
        ret['high1_rate'] = high1_rate
        ret['low1_rate'] = low1_rate
    else: #predict
        ret['highN_rate'] = highN_rate 
        ret['lowN_rate'] = lowN_rate
        ret['high1_rate'] = high1_rate 
        ret['low1_rate'] = low1_rate
    
    # 获取过去所有交易日的min,max, 均值/标准差,z-score?
    # 特征值归一化
    ret["past_days"] = []
    
    df_60 = df.loc[idx:idx+MINMAX_DAYS]
    max_price = df_60['high'].max()
    min_price = df_60['low'].min()
    min_max_rate={}
    for rate in FIELDS:
        min_max_rate[rate]=[df_60[rate].min(),df_60[rate].max()]
    
    df_past = df.loc[idx:idx+PAST_DAYS-1]  
    
    for idx,row in df_past.iterrows(): 
        feather_ret = []
        # 开盘、收盘、最高、最低价，采用过去PAST_DAYS=60天内的最高价、最低价，作为min-max归一化
        for field in FIELDS_PRICE:
            feather_ret.append(minmax(row[field],min_price,max_price))
            
        for field in FIELDS:
            feather_ret.append(minmax(row[field],min_max_rate[field][0],min_max_rate[field][1])) 
        
        #量价关系？
        
        # len(feather_ret)==24    
        ret["past_days"].insert(0,feather_ret) 
        
    # 额外;分割的datestock_uid,current_date,btc_no,dataset_type, 便于后续数据集拆分、pair构造等
    datestock_uid = str(ret["current_date"]) + ret['btc_no']
    if len(ret["past_days"]) == PAST_DAYS:
        li = [str(item) for item in [datestock_uid,ret["current_date"],ret['btc_no'],0,
                                highN_rate,lowN_rate,high1_rate,low1_rate,json.dumps(ret)]] #
        if need_print:
            print(';'.join(li))
        return li
    
    return False

def process_rows(df,start_idx,end_idx):
    all_rows = []
    for idx in range(start_idx, end_idx):
        current_date = int(df.loc[idx]['trade_date'])
        ret = process_row(df, idx,current_date,need_print=False)
        if ret:
            all_rows.append(ret) 
    
    columns="pk_date_btc,trade_date,btc_no,dataset_type,highN_rate,lowN_rate,high1_rate,low1_rate,data_json".split(",")
    df = pd.DataFrame(all_rows,columns=columns)
    df = df.astype({"pk_date_btc":int,"trade_date":int,"btc_no":int,"dataset_type":int,
                    "highN_rate":float,"lowN_rate":float,"high1_rate":float,"low1_rate":float})
    return df

def data_split(df):
    # 分割数据集，train,validate,test。 total=3612, validate/test各300？
    df_test = df.sample(300)
    for idx,rows in df_test.iterrows():
        df.loc[idx,'dataset_type'] = 2 
    
    df_tmp = df[df['dataset_type']==0]
    df_thresholdidate = df_tmp.sample(300)
    for idx,rows in df_thresholdidate.iterrows():
        df.loc[idx,'dataset_type'] = 1 

price_fields = "open,low,high,close".split(",")
def compute_more_rate(df):
    '''1. 相邻交易日，价格，成交量delta值
    2.持有天数与收益情况
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
        
def process():
    df_raw = pd.read_csv(f"data/btc/all_2014_2023.raw.csv",sep=";",header=0,dtype={'trade_date':int})
    
    df_raw = compute_more_rate(df_raw)
    df_raw.to_csv(f"data/btc/all_2014_2023.csv",sep=";",index=None) #header=None,
    
    df = process_rows(df_raw,FUTURE_DAYS,len(df_raw) - PAST_DAYS)
    
    # 分割数据集
    data_split(df)
    df.to_csv("data/btc/all_train_data.csv",sep=";",index=None,header=None)
    
    #统计均值、标准差，2个标准差外的，被一般认为异常值
    # 可对这些异常值人工分析，判断与什么类型的重大实践相关
    static(df)
    

def better(trade_date):
    '''给定日期, 加载60+1天的数据，先计算相邻delta值，再生成训练/预测用到的数值；如果能算出要预测的未来值，则给出
    计算相邻delta值，存在重复计算？提前计算好更稳妥？
    '''
    pass 

def static(df):
    sel_fields = "highN_rate,lowN_rate,high1_rate,low1_rate".split(",")
    df = df[sel_fields]
    d = df.describe()
    print(d)
    total = len(df)
    ret = {}
    for field in sel_fields:
        mean = d[field]['mean']
        std = d[field]['std']
        min_threshold = round(mean - std*2,4)
        max_threshold = round(mean + std*2,4)
        sel_count = len(df[ (df[field]>min_threshold) & (df[field]<max_threshold)])
        print(field,min_threshold,max_threshold,sel_count,round(sel_count*100/total,2))
        ret[field]=[min_threshold,max_threshold]
    print(ret)
    
    '''
    highN_rate    lowN_rate   high1_rate    low1_rate
count  3612.000000  3612.000000  3612.000000  3612.000000
mean      2.969649    -1.511789     2.250323    -2.219405
std       8.221568     7.931796     2.851396     3.002024
min     -31.949900   -49.366200    -7.287800   -38.565500
25%      -1.393750    -5.115500     0.455300    -2.819750
50%       1.772000    -0.953150     1.278900    -1.227900
75%       6.533275     2.618025     2.989600    -0.431650
max      62.198400    35.663000    28.618800     5.538200

highN_rate -13.4735 19.4128 3406 94.3
lowN_rate -17.3754 14.3518 3394 93.96
high1_rate -3.4525 7.9531 3446 95.4
low1_rate -8.2235 3.7846 3447 95.43
    '''

if __name__ == "__main__":
    # python preprocess.py > data/all_train.data
    process()
    # sel_fields = "highN_rate,lowN_rate,high1_rate,low1_rate".split(",")
    # df = pd.read_csv("data/btc/all_train_data.csv",sep=";",header=0,usecols=sel_fields)
    # static(df)