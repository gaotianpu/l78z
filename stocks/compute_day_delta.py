#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import time
import sqlite3  
import json
import logging
import pandas as pd
from common import load_stocks,load_trade_dates,c_round

PROCESSES_NUM = 5

FIELDS_PRICE="OPEN_price;CLOSE_price;LOW_price;HIGH_price".split(";")
FIELDS = "TURNOVER;TURNOVER_amount;TURNOVER_rate".split(";") #change_amount;

conn = sqlite3.connect("file:newdb/stocks.db?mode=ro", uri=True)

def minmax(x,min,max):
    if max == min:
        return 0.00000001
    val = round((x-min)/(max-min),9)
    if val>1:
        val = 1.0
    if val<0:
        val = 0.0
    return val

def zscore(x,mean,std):
    return round((x-mean)/(std+0.00000001),9)

def compute_rate(x,base): #计算涨跌幅度
    return round((x-base)*100/(base+0.00000001),9)

            
def date_statics(df):
    date = df['trade_date'][0]
    d = df.describe()
    
    l = [date]
    for f in ['TURNOVER','TURNOVER_amount']:
        for s in ['mean','std','min','max']:
            l.append(d[f][s])
    return l 

def get_date_static_fields():  
    columns = ['trade_date'] 
    for f in ['TURNOVER','TURNOVER_amount']:
        for s in ['mean','std','min','max']:
            columns.append(f'{f}_{s}')
    # print(columns)
    return  columns    
    
def gen_all_date_statics(start_date=0):
    trade_dates = pd.read_sql(f"select distinct trade_date from stock_raw_daily where trade_date>={start_date}",conn)
    trade_dates = trade_dates['trade_date'].sort_values(ascending=False).tolist()
    
    all_li = []
    for date in trade_dates:
        sql=f"select TURNOVER,TURNOVER_amount from stock_raw_daily where trade_date={date}"
        df = pd.read_sql(sql,conn)
        all_li.append(date_statics(df))
    
    df = pd.DataFrame(all_li,columns=get_date_static_fields())
    df.to_csv(f'data5/date_statics_{start_date}.txt',sep=";",index=None,header=None)

def get_date_statics(start_date=0):
    sql=f"select * from stock_date_statics where trade_date>={start_date}"
    df = pd.read_sql(sql,conn)
    d = {}
    for row_idx,row in df.iterrows():
        d[row["trade_date"]] = row.tolist()
    return d 

def compute_delta(df,date_statics,need_save=True):
    stock_no = df.loc[0]['stock_no']
    # 初始化各种delta字段
    for p1 in FIELDS_PRICE:
        tp1 = p1.replace("_price","")
        for p2 in FIELDS_PRICE:
            tp2 = p2.replace("_price","")
            df[f'delta_{tp1}_{tp2}']=0.0
    for f in FIELDS:
        df[f'delta_{f}']=0.0
    df[f'range_base_lastclose']=0.0
    df[f'range_base_open']=0.0
    df[f'zscore_TURNOVER']=0.0
    df[f'zscore_amount']=0.0
    
    # 计算相邻2日，各个字段的delta值        
    total_cnt = len(df)     
    for idx,row in df.iterrows():
        if (idx+1>=total_cnt): 
            break
        
        current_date = row['trade_date']
        statics = date_statics.get(current_date,False)
        if not statics:
            print("no date_statics:",current_date,statics)
            continue
        
        for p1 in FIELDS_PRICE:
            tp1 = p1.replace("_price","")
            for p2 in FIELDS_PRICE:
                tp2 = p2.replace("_price","")
                df.loc[idx,f'delta_{tp1}_{tp2}'] = compute_rate(row[p1],df.loc[idx+1][p2])
        
        for f in FIELDS:
            # if df.loc[idx+1][f]==0.0:
            #     print(stock_no,f,idx,row['trade_date'])
            df[f'delta_{f}'] = compute_rate(row[f],df.loc[idx+1][f])
        
        if df.loc[idx+1]["CLOSE_price"]>0.0:
            df.loc[idx,f'range_base_lastclose'] = round((row["HIGH_price"] - row["LOW_price"])*100/df.loc[idx+1]["CLOSE_price"],4) #波动幅度
        if df.loc[idx]["OPEN_price"]>0.0:
            df.loc[idx,f'range_base_open'] = round((row["HIGH_price"] - row["LOW_price"])*100/df.loc[idx]["OPEN_price"],4) #波动幅度
        
        TURNOVER_mean = statics[1]
        TURNOVER_std = statics[2]
        df.loc[idx,f'zscore_TURNOVER'] = zscore(row["TURNOVER"],TURNOVER_mean,TURNOVER_std)
        
        amount_mean = statics[5]
        amount_std = statics[6]
        df.loc[idx,f'zscore_amount'] = zscore(row["TURNOVER_amount"],amount_mean,amount_std)
        
    #field count=32: 
    #trade_date;stock_no;OPEN_price;CLOSE_price;LOW_price;HIGH_price;TURNOVER;TURNOVER_amount;TURNOVER_rate;delta_OPEN_OPEN;delta_OPEN_CLOSE;delta_OPEN_LOW;delta_OPEN_HIGH;delta_CLOSE_OPEN;delta_CLOSE_CLOSE;delta_CLOSE_LOW;delta_CLOSE_HIGH;delta_LOW_OPEN;delta_LOW_CLOSE;delta_LOW_LOW;delta_LOW_HIGH;delta_HIGH_OPEN;delta_HIGH_CLOSE;delta_HIGH_LOW;delta_HIGH_HIGH;delta_TURNOVER;delta_TURNOVER_amount;delta_TURNOVER_rate;range_base_lastclose;range_base_open;zscore_TURNOVER;zscore_amount
    if need_save:
        df.to_csv(f"data5/day_delta/{stock_no}.csv",sep=";",index=None,header=None) # 
    return df 

def get_sql_base():
    str_fields = ",".join(FIELDS_PRICE + FIELDS)
    return f"select trade_date,stock_no,{str_fields} from stock_raw_daily" 
    
def process_incremental_delta():
    last_date=20231227 #实际从delta表中取最大值
    
    str_sqlbase = get_sql_base() # ",".join(FIELDS_PRICE + FIELDS)
    sql = f"{str_sqlbase} where trade_date>={last_date} order by stock_no,trade_date desc"
    df = pd.read_sql(sql, conn)
    
    date_statics = get_date_statics(last_date)
    
    df_final = pd.DataFrame() 
    stocks = load_stocks(conn)
    for i, stock in enumerate(stocks):
        stock_no = stock[0]
        df_stock = df[df['stock_no']==stock_no].copy()
        df_stock = df_stock.reset_index(drop=True)
        if len(df_stock)==0:
            print(stock_no)
            continue
        df_tmp = compute_delta(df_stock,date_statics,need_save=False)
        if df_final.empty:
            df_final = df_tmp
        else: 
            df_final = pd.concat([df_final,df_tmp])
    
    df_final = df_final[df_final['trade_date']>last_date]
    df_final = df_final.reset_index(drop=True)
    current_date = df_final['trade_date'][0]
    df_final.to_csv(f"data5/day_delta/d_{current_date}.csv",sep=";",index=None,header=None)

def process_history_delta(processes_idx=-1):
    date_statics = get_date_statics()
    stocks = load_stocks(conn)
    for i, stock in enumerate(stocks):
        if processes_idx < 0 or i % PROCESSES_NUM == processes_idx:   
            stock_no = stock[0] 
            
            str_sqlbase = get_sql_base() # ",".join(FIELDS_PRICE + FIELDS)
            sql = f"{str_sqlbase} where stock_no='{stock_no}' order by trade_date desc"
            df = pd.read_sql(sql, conn)
    
            compute_delta(df,date_statics)
        # break

if __name__ == "__main__":
    data_type = sys.argv[1]
    if data_type == "history":
        # 历史全量
        process_idx = -1 if len(sys.argv) != 3 else int(sys.argv[2])
        process_history_delta(process_idx)
    if data_type == "incremental":
        # 每日增量
        process_incremental_delta()
    if data_type == "date_statics":    
        gen_all_date_statics()
        
    if data_type == "tmp": 
        get_date_statics()