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
    (highN_rate,next_high_rate,next_low_rate) = (0.0,0.0,0.0)
    
    # 未来值,FUTURE_DAYS最高价，最低价？
    if idx>0: #train
        buy_base = df.loc[idx-1]['open'] # df_future.iloc[-1]['TOPEN']
        hold_base = df.loc[idx]['close'] #昨收价格
        
        #保守点，把最高价?作为买入价，用future_days的最低价减去当前的最高价
        highN_rate = compute_rate(df.loc[idx-FUTURE_DAYS]['high'],buy_base)     
        lowN_rate = compute_rate(df.loc[idx-FUTURE_DAYS]['low'],buy_base)     
              
        #用于预测要买入的价位
        next_low_rate = compute_rate(df.loc[idx-1]['low'],hold_base)
        #用于预测要卖出的价位
        next_high_rate = compute_rate(df.loc[idx-1]['high'],hold_base)

        ret['highN_rate'] = highN_rate 
        ret['lowN_rate'] = lowN_rate  #做空机制
        ret['next_high_rate'] = next_high_rate 
        ret['next_low_rate'] = next_low_rate
    else: #predict
        ret['highN_rate'] = highN_rate 
        ret['lowN_rate'] = lowN_rate
        ret['next_high_rate'] = next_high_rate 
        ret['next_low_rate'] = next_low_rate
    
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
                                highN_rate,lowN_rate,next_high_rate,next_low_rate,json.dumps(ret)]] #
        if need_print:
            print(';'.join(li))
        return li
    
    return False

def process():
    df = pd.read_csv(f"data/btc/all_2014_2023.csv",sep=";",header=0,dtype={'trade_date':int})
    end_idx = len(df) - PAST_DAYS
    all_rows = []
    for idx in range(FUTURE_DAYS, end_idx):
        current_date = int(df.loc[idx]['trade_date'])
        ret = process_row(df, idx,current_date,need_print=False)
        if ret:
            all_rows.append(ret) 
    
    columns="pk_date_btc,trade_date,btc_no,dataset_type,highN_rate,lowN_rate,high1_rate,low1_rate,data_json".split(",")
    df = pd.DataFrame(all_rows,columns=columns)
    df = df.astype({"pk_date_btc":int,"trade_date":int,"btc_no":int,"dataset_type":int})
    
    # 分割数据集，train,validate,test。 total=3612, validate/test各300？
    df_test = df.sample(300)
    for idx,rows in df_test.iterrows():
        df.loc[idx,'dataset_type'] = 2 
    
    df_tmp = df[df['dataset_type']==0]
    df_validate = df_tmp.sample(300)
    for idx,rows in df_validate.iterrows():
        df.loc[idx,'dataset_type'] = 1 
    
    df.to_csv("data/btc/all_train_data.csv",sep=";",index=None)

if __name__ == "__main__":
    # python preprocess.py > data/all_train.data
    process()