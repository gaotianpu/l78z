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
# from datetime import datetime, timezone, timedelta
import requests
import re 
import logging
from multiprocessing import Pool
from itertools import islice
import sqlite3  
import pickle

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
    d={}
    for row in data_rows:
        val = p_val_exact.findall(row)
        # print(val,type(val))
        if val:
            val[2] = float(val[2]) #最新价
            val[7] = float(val[7]) #今开盘
            val[8] = float(val[8]) #昨收盘
            val[9] = float(val[9]) #最低价
            val[10] = float(val[10]) #最高价
            
            last = val[8]
            val.append(round((val[9] - last)/last,4)) #11 low_rate
            val.append(round((val[7] - last)/last,4))  #12 open_rate
            val.append( round((val[10] - last)/last,4)) #13 high_rate
            
            d[val[0]] = val  
    # print(page_idx,d)
    return d

def process_all():
    # https://zhuanlan.zhihu.com/p/110005305
    T1 = time.time()
    
    d = {}
    for i in range(102):
        # try:
        ret = process_one_page(i+1)
        if ret:
            d.update(ret)
        # except:
        #     logging.warning("process_all err=%s" % (i))
            
        # break
    
    T2 = time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    
    with open('today.bin', 'wb') as wf:
        pickle.dump(d, wf)
    
    with open ('today.json','w') as f:
        json.dump(d,f, ensure_ascii=False)

def tmp(today_d):
    # select_cols='stock_no,low_rate_25%,low_rate_50%,low_rate_75%,high_rate_25%,high_rate_50%,high_rate_75%'.split(",")
    # df = pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0,dtype={'stock_no': str})
    # d={}
    # for idx,row in df.iterrows():
    #     d[row['stock_no']] = [row['low_rate_25%'],row['high_rate_50%'],row['high_rate_75%']]
    #     # print(row['stock_no'],row['low_rate_25%'],row['high_rate_50%'],row['high_rate_75%'])
    # with open('static_stocks.bin', 'wb') as wf:
    #     pickle.dump(d, wf)
    # return 
    # 获取统计数据
    static_stocks = None
    with open('static_stocks.bin', 'rb') as file:
        static_stocks = pickle.load(file)
    # for k,v in static_stocks.items():
    #     print(k,v) 
    
    df_predict = pd.read_csv("data/predict_merged.txt",sep=";",header=0,dtype={'stock_no': str})
    for idx,row in df_predict.iterrows():
        stock_no = row['stock_no']
        statics = static_stocks.get(stock_no,None)
        today = today_d.get(stock_no)
        print(stock_no,statics,today)
        break 
    return 
        
    x = {'l1':[],'l2':[],'l3':[],'l4':[],'l5':[],'l6':[]}
    for stock_no,val in today_d.items():
        statics = static_stocks.get(stock_no,None)
        if not statics:
            print("not exist in static_stocks.bin:%s,20010101,%s,0"%(stock_no,val[1]))
            continue
        
        last = val[8]
        low_rate = round((val[9] - last)/last,4)
        open_rate = round((val[7] - last)/last,4)  #val[7]
        high_rate = round((val[10] - last)/last,4) #val[10]
        now_rate = round((val[2] - last)/last,4)
        
        # 38 -0.046 low 25% good
        if open_rate < statics[0]: #当前跌幅，比 历史统计的low_rate_25%还低
            print(stock_no,open_rate,statics[0],now_rate,"open low_rate")
            x['l2'].append(now_rate)
        
        # 70 0.0425
        if open_rate > statics[1]: #50%
            print(stock_no,open_rate,statics[1],now_rate,"open high_rate")
            x['l3'].append(now_rate)
        
        # 29 0.0736
        if open_rate > statics[2]: #75%
            print(stock_no,open_rate,statics[1],now_rate,"open high_rate2")
            x['l5'].append(now_rate)
            
        # 231 -0.0272
        if low_rate < statics[0]: #当前跌幅，比 历史统计的low_rate_25%还低
            print(stock_no,low_rate,statics[0],now_rate,"low low_rate")
            x['l1'].append(now_rate) 
        
        # 2023 0.0147
        if high_rate > statics[1]:
            print(stock_no,"high_rate")
            x['l4'].append(now_rate)
        
        #  643 0.0311
        if high_rate > statics[2]:
            print(stock_no,"high_rate2")
            x['l6'].append(now_rate)
        
        # l1: 231 -0.0272
        # l2: 38 -0.046
        # l3: 70 0.0425 
        # l4: 2023 0.0147
        # l5: 29 0.0736
        # l6: 643 0.0311

        for k,v in x.items():
            print(k+":",len(v),round(sum(v)/len(v),4)) 
                   
        # print(round(sum(l1)/len(l1),4),round(sum(l2)/len(l2),4),round(sum(l3)/len(l3),4),round(sum(l4)/len(l4),4),round(sum(l5)/len(l5),4),round(sum(l6)/len(l6),4))
            # print(stock_no,low_rate,open_rate,high_rate,val) 

# uncollect_stock_no.txt
if __name__ == "__main__":
    op_type = sys.argv[1]
    print(op_type)
    if op_type == "all":
        # process_one_page(2)
        process_all()
    if op_type == "tmp":
        with open('today.bin', 'rb') as file:
            today_d = pickle.load(file)
            tmp(today_d)