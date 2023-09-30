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
    

def old():
    # with open('today.bin', 'wb') as wf:
    #     pickle.dump(d, wf)
    
    # with open ('today.json','w') as f:
    #     json.dump(d,f, ensure_ascii=False)
    
    # select_cols='stock_no,low_rate_25%,low_rate_50%,low_rate_75%,high_rate_25%,high_rate_50%,high_rate_75%'.split(",")
    # df = pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0,dtype={'stock_no': str})
    # d={}
    # for idx,row in df.iterrows():
    #     d[row['stock_no']] = [row['low_rate_25%'],row['high_rate_50%'],row['high_rate_75%']]
    #     # print(row['stock_no'],row['low_rate_25%'],row['high_rate_50%'],row['high_rate_75%'])
    # with open('static_stocks.bin', 'wb') as wf:
    #     pickle.dump(d, wf)
    # return 
    # select_cols='stock_no,low_rate_25%,low_rate_50%,low_rate_75%,high_rate_25%,high_rate_50%,high_rate_75%'.split(",")
    # df = pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0,dtype={'stock_no': str})
    # d={}
    # for idx,row in df.iterrows():
    #     d[row['stock_no']] = [row['low_rate_25%'],row['high_rate_50%'],row['high_rate_75%']]
    #     # print(row['stock_no'],row['low_rate_25%'],row['high_rate_50%'],row['high_rate_75%'])
    # with open('static_stocks.bin', 'wb') as wf:
    #     pickle.dump(d, wf)
    # return 
    # select_cols='stock_no,low_rate_25%,low_rate_50%,low_rate_75%,high_rate_25%,high_rate_50%,high_rate_75%'.split(",")
    # df = pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0,dtype={'stock_no': str})
    # d={}
    # for idx,row in df.iterrows():
    #     d[row['stock_no']] = [row['low_rate_25%'],row['high_rate_50%'],row['high_rate_75%']]
    #     # print(row['stock_no'],row['low_rate_25%'],row['high_rate_50%'],row['high_rate_75%'])
    # with open('static_stocks.bin', 'wb') as wf:
    #     pickle.dump(d, wf)
    # return 
    # for idx,row in df.iterrows():
    #     print(row)
    #     break
        # pk_date_stock = str(row['pk_date_stock'])
        # stock_no = pk_date_stock[8:]
        # print(row['stock_no'])
            
        # statics = df_static_stocks[ df_static_stocks['stock_no'] == stock_no ]  #static_stocks.get(stock_no,None)
        # print(stock_no,statics.iloc[0]['high_rate_50%'])
        
        # # if not statics:
        # #     print("statics is none ",stock_no)
        
        # if stock_no[:3]=='688':
        #     print("688:",stock_no,statics,today)
        
        # today = today_d.get(stock_no,None)
        # if today:    
        #     stock_name = today[1]
        #     if 'ST' in stock_name:
        #         print("ST:",stock_no,statics,today)
        # else:
        #     print(stock_no + " not in today_info")
        
        # break 
    # x = {'l1':[],'l2':[],'l3':[],'l4':[],'l5':[],'l6':[]}
    # for stock_no,val in today_d.items():
    #     statics = static_stocks.get(stock_no,None)
    #     if not statics:
    #         print("not exist in static_stocks.bin:%s,20010101,%s,0"%(stock_no,val[1]))
    #         continue
        
    #     last = val[8]
    #     low_rate = round((val[9] - last)/last,4)
    #     open_rate = round((val[7] - last)/last,4)  #val[7]
    #     high_rate = round((val[10] - last)/last,4) #val[10]
    #     now_rate = round((val[2] - last)/last,4)
        
    #     # 38 -0.046 low 25% good
    #     if open_rate < statics[0]: #当前跌幅，比 历史统计的low_rate_25%还低
    #         print(stock_no,open_rate,statics[0],now_rate,"open low_rate")
    #         x['l2'].append(now_rate)
        
    #     # 70 0.0425
    #     if open_rate > statics[1]: #50%
    #         print(stock_no,open_rate,statics[1],now_rate,"open high_rate")
    #         x['l3'].append(now_rate)
        
    #     # 29 0.0736
    #     if open_rate > statics[2]: #75%
    #         print(stock_no,open_rate,statics[1],now_rate,"open high_rate2")
    #         x['l5'].append(now_rate)
            
    #     # 231 -0.0272
    #     if low_rate < statics[0]: #当前跌幅，比 历史统计的low_rate_25%还低
    #         print(stock_no,low_rate,statics[0],now_rate,"low low_rate")
    #         x['l1'].append(now_rate) 
        
    #     # 2023 0.0147
    #     if high_rate > statics[1]:
    #         print(stock_no,"high_rate")
    #         x['l4'].append(now_rate)
        
    #     #  643 0.0311
    #     if high_rate > statics[2]:
    #         print(stock_no,"high_rate2")
    #         x['l6'].append(now_rate)
        
    #     # l1: 231 -0.0272
    #     # l2: 38 -0.046
    #     # l3: 70 0.0425 
    #     # l4: 2023 0.0147
    #     # l5: 29 0.0736
    #     # l6: 643 0.0311

    #     for k,v in x.items():
    #         print(k+":",len(v),round(sum(v)/len(v),4)) 
                   
    #     # print(round(sum(l1)/len(l1),4),round(sum(l2)/len(l2),4),round(sum(l3)/len(l3),4),round(sum(l4)/len(l4),4),round(sum(l5)/len(l5),4),round(sum(l6)/len(l6),4))
    #         # print(stock_no,low_rate,open_rate,high_rate,val) 
    
    # with open('static_stocks.bin', 'rb') as file:
    #     static_stocks = pickle.load(file)
    # for k,v in static_stocks.items():
    #     print(k,v) 
    pass 

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
    
    # 3. 获取统计数据
    df_static_stocks = pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0,dtype={'stock_no': str})
    df_predict = df_predict.merge(df_static_stocks,on="stock_no",how='left')
    
    # 4. 
    select_cols='pk_date_stock,stock_no,pair_high,point_pair_high,point_high,point_low,point_low1,stock_name,last_close,open_rate,low_rate,high_rate,current,low_rate_25%'.split(',')
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