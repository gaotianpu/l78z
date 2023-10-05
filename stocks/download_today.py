#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
今日最新数据下载 - 网易数据源
"""
import os
import sys
import time
import random
import json
import datetime
import pandas as pd
import numpy as np
# from datetime import datetime, timezone, timedelta
import requests
import re 
import logging
from multiprocessing import Pool
from itertools import islice
import sqlite3  
import pickle
from threading import Thread

# https://app.finance.ifeng.com/list/stock.php?t=hs&f=chg_pct&o=desc&p=1

log_file = "log/download_today.log"
logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(lineno)d:%(funcName)s:%(message)s')


def process_one_page(page_idx=102):
    source_url = "https://app.finance.ifeng.com/list/stock.php?t=hs&f=chg_pct&o=desc&p=%s"%(page_idx)
    # resp = requests.get(url=source_url, timeout=0.5)
    
    resp = None
    for i in range(3):  # 失败最多重试3次
        try:
            resp = requests.get(url=source_url, timeout=0.5)
            break
        except:
            if i == 2:  # 超过最大次数
                logging.warning("download fail,retry_times=%s,url=%s" %
                                (i, source_url))
                return False
            else:
                continue
    
    p_val_exact = re.compile(r'>([^>^<]+)<')
    
    start_str = '<table width="914" border="0" cellspacing="0" cellpadding="0">'
    end_str = '<td colspan="11">'

    start_idx = resp.text.find(start_str)
    end_idx =  resp.text.find(end_str)

    # 代码,名称,最新价,涨跌幅↓,涨跌额,成交量,成交额,今开盘,昨收盘,最低价,最高价
    data_txt = resp.text[start_idx+len(start_str):end_idx].replace("\r\n",'').replace(' ','')
    # print(data_txt)
    data_rows = data_txt.replace('<tr>','').split('</tr>')[1:]
    li_ = []
    for row in data_rows:
        val = p_val_exact.findall(row)
        if not val:
            continue
        # print(val,type(val))
        val[2] = float(val[2]) #最新价
        val[3] = float(val[3].replace('%','')) #涨跌幅
        val[5] = int(val[5].replace('手','')) #成交量
        val[6] = int(val[6].replace('万','')) #成交额
        val[7] = float(val[7]) #今开盘
        val[8] = float(val[8]) #昨收盘
        val[9] = float(val[9]) #最低价
        val[10] = float(val[10]) #最高价
        
        last = val[8]
        val.append(round((val[9] - last)/last,3)) #11 low_rate
        val.append(round((val[7] - last)/last,3))  #12 open_rate
        val.append( round((val[10] - last)/last,3)) #13 high_rate
        val.append( round((val[2] - last)/last,3)) #14 current_rate
        
        li_.append(val)
        # d[val[0]] = val  
        
    # print(page_idx,d)
    return li_

def process_all():
    # https://zhuanlan.zhihu.com/p/110005305
    T1 = time.time()
    
    all_li = []
    for i in range(102):
        # try:
        ret = process_one_page(i+1)
        if ret:
            all_li = all_li + ret
            # d.update(ret)
        # except:
        #     logging.warning("process_all err=%s" % (i))
            
        # break
    
    T2 = time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    
    columns = "stock_no,stock_name,current,change_rate,change_price,turnover,amount,open,last_close,low,high,low_rate,open_rate,high_rate,current_rate".split(',')
    df = pd.DataFrame(all_li,columns=columns)
    df.to_csv("data/today.txt",sep=";",index=False) 
    
def tmp(df_today):
    # 1. 获取预测数据
    df_predict = pd.read_csv("data/predict_merged.txt",sep=";",header=0,dtype={'stock_no': str})
    df_predict['stock_no'] = df_predict.apply(lambda x: str(x['pk_date_stock'])[8:] , axis=1)
    # 科创板的暂时先不关注
    df_predict = df_predict[~ df_predict['stock_no'].str.startswith('688')]
    
    # 2. merge今日最新数据
    df_predict = df_predict.merge(df_today,on="stock_no",how='left')
    # ST类型股票不参与
    # df_predict = df_predict[~ df_predict['stock_name'].str.contains('ST')]
    df_predict = df_predict[~ (df_predict['stock_name'].str.contains('ST') | df_predict['stock_no'].str.startswith('688') )]
    
    # 开盘涨停的过滤
    df_predict = df_predict[~ (df_predict["open_rate"]>9.5)]
    # df_predict = df_predict[~ df_predict["change_rate"]<-5]
    
    # 3. 获取统计数据
    df_static_stocks = pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0,dtype={'stock_no': str})
    df_static_stocks_0 = df_static_stocks[df_static_stocks['open_rate_label']==0]
    df_predict = df_predict.merge(df_static_stocks_0,on="stock_no",how='left')
    # (or25,or50,or75) = (df_statics_stock['open_rate_25%'],df_statics_stock['open_rate_50%'],df_statics_stock['open_rate_75%'])
        
    df_predict['current_open_rate_label'] = df_predict.apply(lambda x: 1 if x['open_rate'] < x['open_rate_25%'] else 2 if x['open_rate'] < x['open_rate_50%'] else 3 if x['open_rate'] < x['open_rate_75%'] else 4, axis=1)
    df_predict['buy_prices'] = ''
    df_predict['sell_prices'] = ''
    # 根据open_rate所在区间，计算买入价和卖出价格？
    std_point_low1 = 0.023
    std_point_high1 = 0.023 #待定
    for idx,row in df_predict.iterrows():
        stock_no = row['stock_no']
        stock_static = df_static_stocks[(df_static_stocks['stock_no']==stock_no) & (df_static_stocks['open_rate_label']==row['current_open_rate_label'])]
        
        # point_low1+-,point_high1+-可以移动到 predict_merged.txt 执行？
        low_rates = stock_static.iloc[0][['low_rate_25%','low_rate_50%','low_rate_75%']].values.tolist() 
        low_rates = low_rates + [row['point_low1']-std_point_low1, row['point_low1'], round(row['point_low1'] + std_point_low1,3)]
        buy_prices = (np.array(sorted(low_rates))+1) * row['last_close']
        df_predict.loc[idx, 'buy_prices'] = ','.join([str(v) for v in buy_prices.round(2)]) 
        
        high_rates = stock_static.iloc[0][['high_rate_25%','high_rate_50%','high_rate_75%']].values.tolist()
        high_rates = high_rates + [row['point_high']-std_point_high1, row['point_high'], round(row['point_high'] + std_point_high1,3)]
        sell_prices = (np.array(sorted(high_rates))+1) * row['last_close']
        df_predict.loc[idx, 'sell_prices'] = ','.join([str(v) for v in sell_prices.round(2)])
    
    select_cols='pk_date_stock,stock_no,pair_high,point_pair_high,point_high,last_close,open,open_rate,low,low_rate,high,high_rate,current,buy_prices,sell_prices'.split(',')
    df_predict = df_predict[select_cols]
    
    df_predict.to_csv("data/tmp.txt",sep=";",index=False) 
    

    return 
        
    
# uncollect_stock_no.txt
if __name__ == "__main__":
    op_type = sys.argv[1]
    print(op_type)
    if op_type == "all":
        # process_one_page(2)
        process_all()
    if op_type == "tmp":
        # python download_today.py  tmp 
        df_today = pd.read_csv("data/today.txt",sep=";",header=0,dtype={'stock_no': str,'stock_name': str})
        tmp(df_today)
    
        # today_d = None
        # with open('today.bin', 'rb') as f:
        #     today_d = pickle.load(f) 
        #     tmp(today_d)