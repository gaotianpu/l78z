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
        val.append( round((val[2] - val[7]),3)) #15 current_open
        
        li_.append(val)  
        # d[val[0]] = val  
        
    # print(page_idx,d)
    return li_

def convert_history_format(df):
    #转换为history的格式
    df_ticket_cnt = pd.read_csv("schema/stocks_ticket_cnt.txt",sep=";",header=None, 
                                    names="stock_no,trade_date,TURNOVER,TURNOVER_rate,ticket_cnt".split(","),
                                    dtype={'stock_no': str})
    df_h = df.merge(df_ticket_cnt,on="stock_no",how='inner')
    df_h['TURNOVER_rate'] = round(df_h['turnover']*100 / df_h['ticket_cnt'],2)
    # df_a['TURNOVER_rate'].fillna(0.01) #默认值
    select_cols='stock_no,open,current,change_price,change_rate,low,high,turnover,amount,TURNOVER_rate'.split(",")
    tmp_df = df_h[select_cols]
    
    trade_date=int(datetime.today().strftime("%Y%m%d"))
    tmp_df.insert(loc=0,column="trade_date",value=trade_date)
    tmp_df.to_csv("data/today/convhis_%s.txt"%(trade_date),sep=";",index=False,header=None)
    tmp_df.to_csv("history.data.new",sep=";",index=False,header=None)
    
    
def process_all(last_df,df_predict_v1,df_predict_v2):
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
    
    columns = "stock_no,stock_name,current,change_rate,change_price,turnover,amount,open,last_close,low,high,low_rate,open_rate,high_rate,current_rate,current_open".split(',')
    df = pd.DataFrame(all_li,columns=columns)
    df = df.drop_duplicates(subset=['stock_no'],keep='last')
    
    df["current_rate_delta"] = 0
    if last_df is not None :
        select_cols = "stock_no,current_rate".split(",")
        sel_last_df = last_df[select_cols]
        sel_last_df = sel_last_df.rename(columns={'current_rate':'last_current_rate'})
        df = df.merge(sel_last_df,on="stock_no",how='left')
        df["current_rate_delta"] = round(df["current_rate"] - df["last_current_rate"],4)
         
    trade_time=int(int(datetime.today().strftime("%Y%m%d%H%M"))/10) 
    df.to_csv("data/today/raw_%s.txt"%(trade_time),sep=";",index=False) 
    
    # #转换为history的格式
    # convert_history_format(df)
    
    # 计算买入/卖出价格
    gen_buy_sell_prices(df_predict_v1,df,"v1")
    gen_buy_sell_prices(df_predict_v2,df,"v2")
    return df
    
def gen_buy_sell_prices(df_predict,df_today,version=""):
    trade_date = str(df_predict['pk_date_stock'][0])[:8]
    
    # 科创板的暂时先不关注
    df_predict = df_predict[ (df_predict['stock_no'].str.startswith('688') == False)]
    
    # 2. merge今日最新数据
    df_today = df_today.drop_duplicates(subset=['stock_no'],keep='last')
    df_predict = df_predict.merge(df_today,on="stock_no",how='left')
    # ST类型股票不参与?
    # df_predict = df_predict[~ (df_predict['stock_name'].str.contains('ST')==True)]
    
    # 过滤涨停\跌停股票？
    # 最高最低价相等，视为一种特殊的涨跌停形式
    df_predict = df_predict[(df_predict['high'] - df_predict['low'])>0]
    # 其他的涨停？
    df_predict = df_predict[~ ((df_predict['current_rate']>0.19) & (df_predict['stock_no'].str.startswith('30')))]
    df_predict = df_predict[~ ((df_predict['current_rate']>0.09) & ~(df_predict['stock_no'].str.startswith('30')))]
    
    df_predict['current_open'] = round(df_predict['current_rate'] - df_predict['open_rate'],4)  
    df_predict['current_minmax'] = round((df_predict['current'] - df_predict['low'])/(df_predict['high'] - df_predict['low']),4) 
    
    # 3. 获取统计数据
    df_static_stocks = pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0,dtype={'stock_no': str})
    # df_static_stocks_0 = df_static_stocks[df_static_stocks['open_rate_label']==0]
    # df_predict = df_predict.merge(df_static_stocks_0,on="stock_no",how='left')
    # (or25,or50,or75) = (df_statics_stock['open_rate_25%'],df_statics_stock['open_rate_50%'],df_statics_stock['open_rate_75%'])
       
    df_predict['open_rate_label'] = df_predict.apply(lambda x: 1 if x['open_rate'] < x['open_rate_25%'] else 2 if x['open_rate'] < x['open_rate_50%'] else 3 if x['open_rate'] < x['open_rate_75%'] else 4, axis=1)
    df_predict['in_hold'] = round(df_predict['low_rate'] - df_predict['lowest_rate'],2) 
    
    # df_predict['buy_prices'] = ''
    # df_predict['sell_prices'] = ''
    # # 根据open_rate所在区间，计算买入价和卖出价格？
    # std_point_low1 = 0.023194
    # std_point_high1 = 0.028595 #
    # for idx,row in df_predict.iterrows():
    #     stock_no = row['stock_no']
    #     stock_static = df_static_stocks[(df_static_stocks['stock_no']==stock_no) & (df_static_stocks['open_rate_label']==row['open_rate_label'])]
        
    #     # point_low1+-,point_high1+-可以移动到 predict_merged.txt 执行？
    #     low_rates = stock_static.iloc[0][['low_rate_25%','low_rate_50%','low_rate_75%']].values.tolist() 
    #     low_rates = low_rates + [row['low1.7']-std_point_low1, row['low1.7'], round(row['low1.7'] + std_point_low1,3)]
    #     buy_prices = (np.array(sorted(low_rates))+1) * row['last_close']
    #     df_predict.loc[idx, 'buy_prices'] = ','.join([str(v) for v in buy_prices.round(2)]) 
    #     df_predict.loc[idx,'lowest_rate'] = round(row['low1.7'],4) #round(row['low1.7']-std_point_low1,4)
        
    #     high_rates = stock_static.iloc[0][['high_rate_25%','high_rate_50%','high_rate_75%']].values.tolist()
    #     high_rates = high_rates + [row['point_high1']-std_point_high1, row['point_high1'], round(row['point_high1'] + std_point_high1,3)]
    #     sell_prices = (np.array(sorted(high_rates))+1) * row['last_close']
    #     df_predict.loc[idx, 'sell_prices'] = ','.join([str(v) for v in sell_prices.round(2)])
    
    print(len(df_predict))
    
    
    #df_predict.apply(lambda x: 1 if x['current_rate'] > x['lowest_rate'] else 0)
    # df_predict.to_csv("data/today/predict_%s.txt"%(trade_date),sep=";",index=False)  
    
    
    # sel_fields='pk_date_stock,stock_no,pair_high,point_pair_high,point_high,last_close,open,open_rate,low,low_rate,high,high_rate,current,buy_prices,sell_prices'.split(',')
    # df_predict = df_predict[sel_fields]
    # top3=4,open_rate_label=4 
    # sel_fields = "pk_date_stock,stock_no,open_rate_label,pair_15,point_high1,low1.7,top3,CLOSE_price,LOW_price,HIGH_price,low_rate_std,low_rate_50%,high_rate_std,high_rate_50%,buy_prices,sell_prices".split(",")
    # df_predict = df_predict[(df_predict['top3']==4 & df_predict['open_rate_label']==4 & df_predict['open_rate']<0.1) ]
    # df_predict = df_predict[df_predict['top3']==5]
    # df_predict = df_predict[df_predict['open_rate_label']>=3]
    # df_predict = df_predict[df_predict['open_rate']<0.09] #涨停的不考虑
    # df_predict = df_predict[df_predict['current_open']>0]
    
    # df_predict = df_predict.sort_values(by=["pair_15"],ascending=False)
    # sel_fields = "pk_date_stock,stock_no,CLOSE_price,open_rate,low_rate,high_rate,current_rate,current_open".split(",")
    # df_predict[sel_fields].to_csv("predict_today_show.txt",sep=";",index=False)
    
    # 生成html数据
    sel_fields = "stock_no,stock_name,in_hold,open_rate_label,current_rate,current_minmax,current_rate_delta,lowest_rate".split(",")
    df_html = df_predict[sel_fields]
    html_li = []
    html_li.append("<head>%s</head>" % (datetime.today().strftime("%Y%m%d %H%M")))
    html_li.append("<table>")
    html_li.append("<tr>" + "".join(["<td>%s</td>"%(f) for f in sel_fields]) + "</tr>")
    for idx,row in df_html.iterrows():
        tr_color = "" 
        if row['current_minmax']>0.55: #and row['current_rate_delta']>0
            tr_color = 'style="color:red"'
        
        columns = []
        for field in sel_fields:
            color = ""
            if field == "in_hold" and row['in_hold'] < 0:
                color = 'style="background-color:green"'
            if field == "current_rate_delta" and row['current_rate_delta'] < -0.005:
                color = 'style="background-color:green"'
            if field == "current_minmax" and row['current_minmax']<0.45:
                color = 'style="background-color:green"'
            if field == "current_rate" and row['current_rate']<0:
                color = 'style="background-color:green"'
            columns.append("<td %s>%s</td>"%(color,row[field]))
        html_li.append("<tr %s>%s</tr>" %(tr_color,"".join(columns)))
    html_li.append("</table>")
    
    with open('predict_today_show_%s.html'%(version),'w') as f:
        f.writelines(html_li)
        f.close()

def run_no_stop():
    df_predict_v1 = pd.read_csv("data/predict/predict_merged.txt",sep=";",header=0,dtype={'stock_no': str})
    # 只关注预测结果top3=5部分的数据
    df_predict_v1 = df_predict_v1[df_predict_v1['top3']==5]
    df_predict_v1['lowest_rate'] = df_predict_v1['low1.7']
    
    df_predict_v2 = pd.read_csv("data/predict_v2/predict_merged.txt",sep=";",header=0,dtype={'stock_no': str})
    # 只关注预测结果top3=3部分的数据
    df_predict_v2 = df_predict_v2[df_predict_v2['top3']==3]
    df_predict_v2['lowest_rate'] = df_predict_v2['point_low1']
    
    last_df = None 
    last_file = "data/today/raw_%s.txt" % (int(int((datetime.now()- timedelta(minutes=10)).strftime("%Y%m%d%H%M"))/10))
    if os.path.exists(last_file):
        last_df = pd.read_csv(last_file,sep=";",header=0,dtype={'stock_no': str})
    
    last_trade_time = 0
    while True:
        Hm = int(datetime.today().strftime("%H%M"))
        if (Hm > 935  and Hm < 1140) or (Hm>1300 and Hm<1510): 
            trade_time=int(int(datetime.today().strftime("%Y%m%d%H%M"))/10)
            if trade_time != last_trade_time:
                print(trade_time)
                last_df = process_all(last_df,df_predict_v1,df_predict_v2)
                
            last_trade_time = trade_time    
        time.sleep(60)


#  python download_today.py all
if __name__ == "__main__":
    op_type = sys.argv[1]
    print(op_type)
    if op_type == "all":
        # process_one_page(2)
        process_all()
    if op_type == "gen_buy_sell_prices":
        # python download_today.py  gen_buy_sell_prices 
        df_today = pd.read_csv("data/today/raw_20231030150.txt",sep=";",header=0,dtype={'stock_no': str,'stock_name': str})
        
        df_predict = pd.read_csv("data/predict/predict_merged.txt",sep=";",header=0,dtype={'stock_no': str})
        # 只关注预测结果top3=5部分的数据
        df_predict = df_predict[df_predict['top3']==5]
        df_predict['lowest_rate'] = df_predict['low1.7']
        gen_buy_sell_prices(df_predict,df_today,"v1")
        
        df_predict = pd.read_csv("data/predict_v2/predict_merged.txt",sep=";",header=0,dtype={'stock_no': str})
        # 只关注预测结果top3=5部分的数据
        df_predict = df_predict[df_predict['top3']==3]
        df_predict['lowest_rate'] = df_predict['point_low1']
        gen_buy_sell_prices(df_predict,df_today,"v2")
        
    if op_type == "no_stop":
        run_no_stop()