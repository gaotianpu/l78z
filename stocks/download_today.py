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
        val.append(round((val[9] - last)/last,4)) #11 low_rate
        val.append(round((val[7] - last)/last,4))  #12 open_rate
        val.append( round((val[10] - last)/last,4)) #13 high_rate
        val.append( round((val[2] - last)/last,4)) #14 rate_now
        val.append( round((val[2] - val[7]),4)) #15 current_open
        
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

def after_download(last_df):  
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
    
    columns = "stock_no,stock_name,current,change_rate,change_price,turnover,amount,open,last_close,low,high,low_rate,open_rate,high_rate,rate_now,current_open".split(',')
    df = pd.DataFrame(all_li,columns=columns)
    df = df.drop_duplicates(subset=['stock_no'],keep='last') 

    df["rate_delta"] = 0
    if last_df is not None :
        select_cols = "stock_no,rate_now".split(",")
        sel_last_df = last_df[select_cols]
        sel_last_df = sel_last_df.rename(columns={'rate_now':'last_rate'})
        df = df.merge(sel_last_df,on="stock_no",how='left')
        df["rate_delta"] = round(df["rate_now"] - df["last_rate"],4)
    
    trade_time=int(int(datetime.today().strftime("%Y%m%d%H%M"))/10) 
    cache_file = "data/today/raw_%s.txt"%(trade_time)
    print(cache_file)
    df.to_csv(cache_file,sep=";",index=False)  
    return df 
    
def process_all(last_df,df_predict,one_time=False):
    df = None
    if one_time:
        print("cache shoot")
        today = datetime.today().strftime("%Y%m%d")
        cache_file = f"data/today/raw_{today}150.txt"
        # cache_file = f"data/today/raw_20231215150.txt"
        df = pd.read_csv(cache_file,sep=";",header=0,dtype={'stock_no': str,'stock_name': str})
        df["rate_now"] = df["last_rate"]
    else:
        df = after_download(last_df)
    
    # #转换为history的格式
    # convert_history_format(df)
    
    # 计算买入/卖出价格
    gen_buy_sell_prices(df,df_predict,"v1")
    gen_buy_sell_prices(df,df_predict,"v1","open_rate")
    
    # tmp_df = df_predict_v2[df_predict_v2['top3']==4] 
    # gen_buy_sell_prices(df,tmp_df,"v2")
    # # gen_buy_sell_prices(df,df_predict_v12,"v1_2")
    
    # tmp_df = df_predict_v3[df_predict_v3['top3']>0]
    # # tmp_df = df_predict_v3[df_predict_v3['top5']>7]
    # tmp_df = tmp_df.sort_values(by=["top3","list_1"],ascending=False)
    # gen_buy_sell_prices(df,tmp_df,"v3")
    # gen_buy_sell_prices(df,tmp_df,"v3","open_rate")
    
    # # 持有部分
    # hold_stocks = []
    # with open('hold_stocks.txt','r') as f:
    #     hold_stocks = [row.strip().split(',')[0] for row in f.readlines()]
    # df_holds = df_predict_v3[df_predict_v3['stock_no'].isin(hold_stocks)]
    # df_holds = df_holds.sort_values(by=["top3","pair_date"],ascending=False)
    # gen_buy_sell_prices(df,df_holds,"holds")
    
    return df
    
def gen_buy_sell_prices(df_today,df_predict,version="",sort_field=""):
    trade_date = str(df_predict['pk_date_stock'].values[0])[:8]
    
    # 科创板的暂时先不关注
    df_predict = df_predict[ (df_predict['stock_no'].str.startswith('688') == False)]
    
    # 2. merge今日最新数据
    df_today = df_today.drop_duplicates(subset=['stock_no'],keep='last')
    df_predict = df_predict.merge(df_today,on="stock_no",how='left')
    # ST类型股票不参与?
    # df_predict = df_predict[~ (df_predict['stock_name'].str.contains('ST')==True)]
    
    if sort_field:
        # df_predict = df_predict[df_predict['open_rate']>0]
        df_predict = df_predict.sort_values(by=[sort_field],ascending=False)
    
    # 过滤涨停\跌停股票？
    # 最高最低价相等，视为一种特殊的涨跌停形式
    # df_predict = df_predict[(df_predict['high'] - df_predict['low'])>0]
    # # 其他的涨停？
    # df_predict = df_predict[~ ((df_predict['rate_now']>0.19) & (df_predict['stock_no'].str.startswith('30')))]
    # df_predict = df_predict[~ ((df_predict['rate_now']>0.09) & ~(df_predict['stock_no'].str.startswith('30')))]
    
    df_predict['current_open'] = round(df_predict['rate_now'] - df_predict['open_rate'],4)  
    df_predict['rate_minmax'] = round((df_predict['current'] - df_predict['low'])/(df_predict['high'] - df_predict['low']),4) 
    # df_predict['open_rate_label'] = df_predict.apply(lambda x: 1 if x['open_rate'] < x['open_rate_25%'] else 2 if x['open_rate'] < x['open_rate_50%'] else 3 if x['open_rate'] < x['open_rate_75%'] else 4, axis=1)
    df_predict['in_hold'] = round(df_predict['rate_now'] - df_predict['point_low1'],4)
    df_predict['in_hold_2'] = round(df_predict['rate_now'] - df_predict['point_high1'],4)
    
    df_predict['lowest'] = round(df_predict['last_close'] * (1+df_predict['point_low1']),2)
    df_predict['high1_price'] = round(df_predict['last_close'] * (1+df_predict['point_high1']),2)
    
    
    # 3. 获取统计数据
    # df_static_stocks = pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0,dtype={'stock_no': str})
    # df_static_stocks_0 = df_static_stocks[df_static_stocks['open_rate_label']==0]
    # df_predict = df_predict.merge(df_static_stocks_0,on="stock_no",how='left')
    # (or25,or50,or75) = (df_statics_stock['open_rate_25%'],df_statics_stock['open_rate_50%'],df_statics_stock['open_rate_75%'])
    
    good_cnt = len(df_predict[df_predict['open_rate']>0])
    
    # 生成html数据
    # open_rate_label,cls3_idx,
    sel_fields = "stock_no,stock_name,open_rate,rate_now,rate_minmax,rate_delta,low_rate,low,high,high_rate,cls3,cls3_0,cls3_2,point_low1,point_high1,buy_prices,sell_prices".split(",")
    df_html = df_predict[sel_fields]
    html_li = []
    html_li.append("<head>%s, count=%s %s</head>" % (datetime.today().strftime("%Y%m%d %H%M"),len(df_predict),good_cnt))
    html_li.append("<table>")
    html_li.append("<tr><td>idx</td>" + "".join(["<td>%s</td>"%(f) for f in sel_fields]) + "</tr>")
    for idx,row in df_html.iterrows():
        tr_color = "" 
        if row['rate_now']>0.01:  #row['rate_minmax']>0.5 and row['rate_delta']>=0:
            # tr_color = 'style="color:red"'
            pass

        columns = []
        columns.append("<td>%s</td>"%(idx))
        for field in sel_fields:
            color = "None"
            if field == "buy_prices":
                li = []
                for x in row['buy_prices'].split(','):
                    if row["low"] < float(x) :
                        li.append('<font style="background-color:%s">%s</font>'%('#FFA500',x))
                    else:
                        li.append(x)
                row['buy_prices'] = ",".join(li)
            if field == "sell_prices":
                li = []
                for x in row['sell_prices'].split(','):
                    if row["high"] >= float(x) :
                        li.append('<font style="background-color:%s">%s</font>'%('#FFA500',x))
                    else:
                        li.append(x)
                row['sell_prices'] = ",".join(li)
            if field == "in_hold" : 
                if row['in_hold'] < -0.0215:
                    color = 'green'
                if row["low_rate"]<row["point_low1"]:
                    color = '#FFA500'
            if field == "rate_delta":
                if row['rate_delta'] < -0.001:
                    color = 'green'
                if row['rate_delta'] > 0.001:
                    color = '#FFA500'
            if field == "rate_minmax": 
                if row['rate_minmax']<0.35:
                    color = 'green'
                if row['rate_minmax']>0.55:
                    color = '#FFA500'
            if field == "rate_now": 
                if row['rate_now']<-0.03:
                    color = 'green'
                if row['rate_now']>0.025:
                    color = '#FFA500'
            if field == "point_low1":
                if row["low_rate"]<row["point_low1"]:
                    color = 'green'   
            if field == "stock_name":
                if row["high_rate"] < 0 :
                    color = 'green'
            if field == "high_rate":
                if row["high_rate"]>(row["point_high1"]-0.02): #0.0151
                    color = '#FFA500'
            columns.append('<td style="background-color:%s">%s</td>'%(color,row[field]))
        html_li.append("<tr %s>%s</tr>\n" %(tr_color,"".join(columns)))
    html_li.append("</table>")
    
    with open(f'predict_today_show_{version}{sort_field}.html','w') as f:
        f.writelines(html_li)
        f.close()
    
    Hm = int(datetime.today().strftime("%H%M"))
    if Hm>1455 and Hm<1550:
        today = datetime.today().strftime("%Y%m%d") 
        fname = f'data/today/predict_today_show_{version}{sort_field}_{today}.html'
        print(fname)
        with open(fname,'w') as f:
            f.writelines(html_li)
            f.close()

def run_no_stop(one_time=False):
    df_predict = pd.read_csv("data4/predict/predict_merged.txt",sep=";",header=0,dtype={'stock_no': str})
    df_predict = df_predict[df_predict['cls3_idx']==1]
    
    # df_predict_v2 = pd.read_csv("data/predict_v2/predict_merged.txt",sep=";",header=0,dtype={'stock_no': str})
    # df_predict_v2 = df_predict_v2.sort_values(by=["top3","point2pair_dates"],ascending=False)
    
    # df_predict_v12 = pd.read_csv("data/predict_v2/predict_merged_v1_2.txt",sep=";",header=0,dtype={'stock_no': str})
    # df_predict_v12 = df_predict_v12[df_predict_v12['top3']>7]
    
    # df_predict_v3 = pd.read_csv("data3/predict/predict_merged.txt",sep=";",header=0,dtype={'stock_no': str})
    # df_predict_v3 = df_predict_v3.sort_values(by=["top3","point_high"],ascending=False)
    
    last_df = None 
    last_file = "data/today/raw_%s.txt" % (int(int((datetime.now()- timedelta(minutes=10)).strftime("%Y%m%d%H%M"))/10))
    if os.path.exists(last_file):
        last_df = pd.read_csv(last_file,sep=";",header=0,dtype={'stock_no': str})
    
    if one_time:
        # last_df = pd.read_csv("data/today/raw_20231215150.txt",sep=";",header=0,dtype={'stock_no': str})
        process_all(last_df,df_predict,one_time)
        return 
    
    last_trade_time = 0
    while True:
        Hm = int(datetime.today().strftime("%H%M"))
        if (Hm > 931  and Hm < 1140) or (Hm>1305 and Hm<1510): 
            trade_time=int(int(datetime.today().strftime("%Y%m%d%H%M"))/10)
            if trade_time != last_trade_time:
                print(trade_time)
                last_df = process_all(last_df,df_predict,one_time)
            last_trade_time = trade_time
        time.sleep(60)


#  python download_today.py all
if __name__ == "__main__":
    op_type = sys.argv[1]
    print(op_type)
    if op_type == "all":
        # process_one_page(2)
        process_all()
    if op_type == "one_time":
        # python download_today.py  one_time
        run_no_stop(one_time = True)
    if op_type == "no_stop":
        run_no_stop()
        
    if op_type == "gen_buy_sell_prices":
        # python download_today.py  gen_buy_sell_prices 
        df_today = pd.read_csv("data/today/raw_20231030150.txt",sep=";",header=0,dtype={'stock_no': str,'stock_name': str})
        
        # df_predict = pd.read_csv("data/predict/predict_merged.txt",sep=";",header=0,dtype={'stock_no': str})
        # # 只关注预测结果top3=5部分的数据
        # df_predict = df_predict[df_predict['top3']==5] 
        # gen_buy_sell_prices(df_today,df_predict,"v1")
        
        df_predict_v2 = pd.read_csv("data/predict_v2/predict_merged.txt",sep=";",header=0,dtype={'stock_no': str})
        # # 只关注预测结果top3=5部分的数据
        # df_predict = df_predict[df_predict['top3']==3]
        # gen_buy_sell_prices(df_today,df_predict,"v2")
        
        # df_predict = pd.read_csv("data/predict_v2/predict_merged_v1_2.txt",sep=";",header=0,dtype={'stock_no': str})
        # 只关注预测结果top3=5部分的数据
        # 
        # pair_15,list_dates, point2pair_dates
        # df_predict = df_predict.sort_values(by=["top3","point2pair_dates"],ascending=False)
        # df_predict = df_predict.sort_values(by=["point2pair_dates_top5","list_dates_top5"],ascending=False)
        # df_predict = df_predict[df_predict['top3']>7]
        # gen_buy_sell_prices(df_today,df_predict,"v1_2")