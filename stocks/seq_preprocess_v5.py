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

FUTURE_DAYS = 5 # 预测未来几天的数据, 2,3,5? 2比较合适，3则可能出现重复，在不恰当的数据集划分策略下，训练集和测试可能会重叠？
PAST_DAYS = 20 #使用过去几天的数据做特征
MINMAX_DAYS = 60 

MIN_TRADE_DATE = 0 #20230825

conn = sqlite3.connect("file:newdb/stocks.db?mode=ro", uri=True)

FIELDS_PRICE="OPEN_price;CLOSE_price;LOW_price;HIGH_price".split(";")
FIELDS = "TURNOVER;TURNOVER_amount;TURNOVER_rate".split(";") #change_amount;
FIELDS_DELTA="delta_OPEN_OPEN,delta_OPEN_CLOSE,delta_OPEN_LOW,delta_OPEN_HIGH,delta_CLOSE_OPEN,delta_CLOSE_CLOSE,delta_CLOSE_LOW,delta_CLOSE_HIGH,delta_LOW_OPEN,delta_LOW_CLOSE,delta_LOW_LOW,delta_LOW_HIGH,delta_HIGH_OPEN,delta_HIGH_CLOSE,delta_HIGH_LOW,delta_HIGH_HIGH,delta_TURNOVER,delta_TURNOVER_amount,delta_TURNOVER_rate,range_base_lastclose,range_base_open".split(",")


log_file = "log/seq_preprocess.log"
logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(lineno)d:%(funcName)s:%(message)s')

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

def process_row(df, idx):
    current_date = int(df.loc[idx]['trade_date'])
    stock_no = df.loc[idx]['stock_no']
    ret = {"stock_no": stock_no, "current_date":current_date}
    
    (highN_rate,next_high_rate,next_low_rate) = (0.0,0.0,0.0)
    
    # 未来值,FUTURE_DAYS 最高价，最低价？
    if idx>0: #train
        #保守点，把最高价?作为买入价，用future_days的最低价减去当前的最高价
        high_price = df.loc[idx-1]['HIGH_price'] # df.iloc[idx-1]['OPEN_price']
        low_price = df.loc[idx-1]['LOW_price']
        
        # 如果遇到第二天整天处于涨跌停状态的，也不考虑进入训练数据(因为无法买入？)
        if high_price == low_price:
            print("high=low",stock_no,current_date)
            return False
        
        highN_rate = compute_rate(df.loc[idx-FUTURE_DAYS]['HIGH_price'],high_price)   
        
        hold_base = df.loc[idx]['CLOSE_price'] #昨收价格        
        #用于预测要买入的价位
        next_low_rate = compute_rate(df.loc[idx-1]['LOW_price'],hold_base)
        #用于预测要卖出的价位
        next_high_rate = compute_rate(df.loc[idx-1]['HIGH_price'],hold_base)
        
        # (t+1的最高价-t的最高价)/t的最高价
        
        #是否会跌停的判断？还没想清楚

        ret['highN_rate'] = highN_rate 
        ret['next_high_rate'] = next_high_rate 
        ret['next_low_rate'] = next_low_rate
        ret['val_label'] = 0 
    else: #predict
        ret['highN_rate'] = highN_rate 
        ret['next_high_rate'] = next_high_rate 
        ret['next_low_rate'] = next_low_rate
        ret['val_label'] = 0 
    
    # 获取过去所有交易日的均值，标准差
    # 只能是过去的，不能看到未来数据？ 还是说应该固定住？似乎应该固定住更好些 
    
    # 特征值归一化
    ret["past_days"] = []
    
    df_60 = df.loc[idx:idx+MINMAX_DAYS]
    max_price = df_60['HIGH_price'].max()
    min_price = df_60['LOW_price'].min()
    min_max_rate={}
    for rate in FIELDS+FIELDS_DELTA:
        min_max_rate[rate]=[df_60[rate].min(),df_60[rate].max()]
    
    df_past = df.loc[idx:idx+PAST_DAYS-1] 
    
    for row_idx,row in df_past.iterrows(): 
        feather_ret = []
        for field in FIELDS_PRICE:
            feather_ret.append(minmax(row[field],min_price,max_price))
            
        for field in FIELDS+FIELDS_DELTA:
            feather_ret.append(minmax(row[field],min_max_rate[field][0],min_max_rate[field][1])) 
        
        # 当日成交金额的全局对比，哪些是热点股票
        # feather_ret.append(minmax(row["TURNOVER_amount"],0.01,4796717.0)) 
        # feather_ret.append(minmax(row["TURNOVER"],0.0,41144528.0))
        
        ret["past_days"].insert(0,feather_ret) 
        
    # 额外;分割的datestock_uid,current_date,stock_no,dataset_type, 便于后续数据集拆分、pair构造等
    datestock_uid = str(ret["current_date"]) + ret['stock_no']
    if len(ret["past_days"]) == PAST_DAYS:
        li = [str(item) for item in [datestock_uid,ret["current_date"],ret['stock_no'],0,
                                highN_rate,next_high_rate,next_low_rate,
                                ret['val_label'],json.dumps(ret)]] #
        return li
    
    return False
    
def compute_seq_dates(df,stock_no):
    dates = []
    end_idx = len(df) - PAST_DAYS - 1
    for idx in range(FUTURE_DAYS, end_idx): 
        # 计算过去、未来的时间间隔，用于将包含停牌期间的数据排除
        current_date = int(df.loc[idx]['trade_date'])
        future_date = df.loc[idx-FUTURE_DAYS]['trade_date']
        past_date = df.loc[idx+PAST_DAYS]['trade_date']
        
        date0 = datetime.strptime(str(past_date), "%Y%m%d").date() 
        date1 = datetime.strptime(str(current_date), "%Y%m%d").date()
        date2 = datetime.strptime(str(future_date), "%Y%m%d").date() 
            
        past_days = (date1 - date0).days
        future_days = (date2 - date1).days
        # future_days->5+2, past_days:4周*7+2, 例外情况春节和国情，额外7(+4)天长假
        # 间隔较长的拎出来，看下是否异常
        if future_days>18 or past_days>41:
            dates.append([stock_no,current_date,past_date,future_date,past_days,future_days])
    
    dates_columns = "stock_no,current_date,past_date,future_date,past_days,future_days".split(",")
    df_dates = pd.DataFrame(dates,columns=dates_columns)
    df_dates.to_csv(f'data5/seq_dates/{stock_no}.csv',sep=";",index=None)

def gen_train_data(df,stock_no):
    all_li = []
    end_idx = len(df) - PAST_DAYS - 1
    for idx in range(FUTURE_DAYS, end_idx): 
        # 根据时间间隔，将包含停牌期间的数据排除
        current_date = int(df.loc[idx]['trade_date'])
        future_date = df.loc[idx-FUTURE_DAYS]['trade_date']
        past_date = df.loc[idx+PAST_DAYS]['trade_date']
        
        date0 = datetime.strptime(str(past_date), "%Y%m%d").date() 
        date1 = datetime.strptime(str(current_date), "%Y%m%d").date()
        date2 = datetime.strptime(str(future_date), "%Y%m%d").date() 
            
        past_days = (date1 - date0).days
        future_days = (date2 - date1).days 
        
        # future_days->5+2, past_days:4周*7+2, 例外情况春节和国情，额外7(+4)天长假
        if future_days>18 or past_days>41:
            continue

        ret = process_row(df, idx)
        if ret :
            all_li.append(ret)  

    columns = "pk_date_stock;trade_date;stock_no;dataset_type;highN_rate;high1_rate;low1_rate;list_label;data_json".split(";")
    df_seq = pd.DataFrame(all_li,columns=columns)
    df_seq.to_csv(f'data5/seq_train/{stock_no}.csv',sep=";",index=None,header=None)
                  
def process_train_data(processes_idx=-1):
    stocks = load_stocks(conn)
    for i, stock in enumerate(stocks):
        if processes_idx < 0 or i % PROCESSES_NUM == processes_idx:   
            stock_no = stock[0]  
            sql = f"select * from stock_with_delta_daily where stock_no='{stock_no}' order by trade_date desc"
            df = pd.read_sql(sql, conn) 
            
            gen_train_data(df,stock_no)
        break

def process_predict_data(current_date=None):
    all_li = []
    stocks = load_stocks(conn)
    limit_cnt = MINMAX_DAYS + 1
    for i, stock in enumerate(stocks):
        stock_no = stock[0] 
        sql = f"select * from stock_with_delta_daily where stock_no='{stock_no}' order by trade_date desc limit {limit_cnt}"
        df = pd.read_sql(sql, conn)
        ret = process_row(df, 0)
        if ret:
            all_li.append(ret)
            
    columns = "pk_date_stock;trade_date;stock_no;dataset_type;highN_rate;high1_rate;low1_rate;list_label;data_json".split(";")
    df = pd.DataFrame(all_li,columns=columns)
    df = df.sort_values(by=['highN_rate'],ascending=False)
    # trade_date = df['trade_date'].max()
    df.to_csv(f"data5/seq_predict/{current_date}.data",sep=";",header=None,index=None)


if __name__ == "__main__":
    data_type = sys.argv[1]
    if data_type == "train":
        process_idx = -1 if len(sys.argv) != 3 else int(sys.argv[2])
        process_train_data(process_idx)
    
    if data_type == "predict" :
        process_predict_data()