#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import time
import sqlite3  
import pandas as pd
import json
from common import load_stocks,get_max_trade_date,load_trade_dates

MAX_ROWS_COUNT = 2000 #从数据库中加载多少数据, 差不多8年的交易日数。

forecast_fields = 'f_high_mean_rate,f_high_rate,f_low_mean_rate,f_low_rate'.split(",")
static_fields = 'mean,std,min,25%,50%,75%,max'.split(",")

def get_all_fields():
    fields = ['key','count']
    for f1 in forecast_fields:
        for f2 in static_fields:
            fields.append("%s_%s"%(f1,f2))
    return fields

conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)

#  select * from stock_raw_daily_2 order by RANDOM() limit 2;
# select stock_no,round(avg(TURNOVER_rate),2) from stock_raw_daily_2 where stock_no in (300879,300880,300881,300882,300883,300884,300885,300886,300887,300889,300892,300893,300894,300895,300896,300897,300898,300899,300900,300901,300902,300910,300911,300912,300913,300915,300916,300919,300925,300928,300929,300935,300939,300940,300946,300948,300949,300951,300952,300953,300958,300959,300960,300961,300968,300971,300973,300999,301000,301001) and  TURNOVER_rate<>'-' group by stock_no;
# update stock_raw_daily_2 set TURNOVER_rate=%s where stock_no='%s' and TURNOVER_rate='-';
def c_round(x):
    return round(x,4)

def get_update_sql():
    sql = "select stock_no,round(avg(TURNOVER_rate),2) as TURNOVER_rate from stock_raw_daily_2 where stock_no in ('000596','000796','001914','002074','002789','002798','002801','002809','002819','002820','002821','002840','002893','002899','002900','002901','002923','002926','002937','002939','002950','002952','002953','002959','002961','002962','002963','002965','002966','002973','002976','002984','002987','002988','002990','002991','002992','002993','002997','002999','003000','003003','003006','003011','003012','003015','003016','003017','003018','003019','003026','003027','003030','003035','003036','003037','003040','003816') and  TURNOVER_rate<>'-' group by stock_no;"
    df = pd.read_sql(sql, conn)
    # df.loc[idx]['trade_date']
    for index, row in df.iterrows():
        usql = ("update stock_raw_daily_2 set TURNOVER_rate=%s where stock_no='%s' and TURNOVER_rate='-';" %(row["TURNOVER_rate"],row["stock_no"]))
        print(usql)

def compute_mean_std(stock_no=None):
    '''抽样计算整体的均值和标准差'''
    # order by RANDOM()
    fields = "OPEN_price,CLOSE_price,change_amount,change_rate,LOW_price,HIGH_price,TURNOVER,TURNOVER_amount,TURNOVER_rate,low_rate".split(",")
    
    ret = {}
    sql = "select * from stock_raw_daily_2  order by RANDOM() limit 350000;"
    if stock_no:
        ret["stock_no"] = stock_no
        sql = "select * from stock_raw_daily_2 where stock_no='%s' order by trade_date desc"%(stock_no)
    
    df = pd.read_sql(sql, conn)
    
    # if stock_no=="002913":
    #     df['change_rate'] = df['change_rate'].str.replace("%","")
    #     df['change_rate'] = pd.to_numeric(df['change_rate']) #.astype('double')
    #     df['TURNOVER_rate'] = df['TURNOVER_rate'].str.replace("%","")
    #     df['TURNOVER_rate'] = pd.to_numeric(df['TURNOVER_rate'])
    
    df['last_close_price'] = df['CLOSE_price'] - df['change_amount'] 
    df['low_rate'] = c_round((df['LOW_price'] - df['last_close_price']) / df['last_close_price']) 
    
    # print(df.head())
    # print(df['low_rate'].head())
    df_describe = df.describe() 
    # print(df_describe) 
    
    for field in fields:
        for f2 in static_fields:
            ret["%s_%s" %(field,f2)] = c_round(df_describe[field][f2])
        #ret[field+"_std"] = c_round(df_describe[field]["std"])
        # 中位数，75%，25%？
    
    if stock_no:
        max_trade_date = get_max_trade_date(conn,stock_no)
        print("%s;%s;%s" %(stock_no, max_trade_date,json.dumps(ret)))
    else:
        print(json.dumps(ret))
        

def compute_per_stocks_mean_std():
    '''计算每个stocks，价格，成交量，成交金额等字段的均值、标准差'''
    stocks = load_stocks(conn)
    time_start = time.time()
    for i, stock in enumerate(stocks):
        stock_no = stock[0]
        try:
            compute_mean_std(stock_no)
        except:
            print("error:",stock_no)
        # break

def tmp():
    stocks = load_stocks(conn)
    time_start = time.time()
    for i, stock in enumerate(stocks):
        stock_no = stock[0]
        sql = "select * from stock_raw_daily where stock_no='%s' and TOPEN>0 order by trade_date desc limit 0,%d"%(stock_no,MAX_ROWS_COUNT)
        df = pd.read_sql(sql, conn)
        
        last_trade_date = df["trade_date"].max()
        
        # 统计价格、成交量、成交金额、换手率等的均值和标准差，用于后续的归一化处理
        df_history_describe = df.describe() 
        # 价格
        price_mean = c_round(df_history_describe["TOPEN"]["mean"])
        price_std = c_round(df_history_describe["TOPEN"]["std"]) 
        # VOTURNOVER 成交金额
        VOTURNOVER_mean = c_round(df_history_describe["VOTURNOVER"]["mean"])
        VOTURNOVER_std = c_round(df_history_describe["VOTURNOVER"]["std"])
        # VATURNOVER 成交量
        VATURNOVER_mean = c_round(df_history_describe["VATURNOVER"]["mean"])
        VATURNOVER_std = c_round(df_history_describe["VATURNOVER"]["std"])
        #  ('TURNOVER', '换手率')
        TURNOVER_mean = c_round(df_history_describe["TURNOVER"]["mean"])
        TURNOVER_std = c_round(df_history_describe["TURNOVER"]["std"]) 
        
        ret ={"price_mean":price_mean,"price_std":price_std,"VOTURNOVER_mean":VOTURNOVER_mean,"VOTURNOVER_std":VOTURNOVER_std,
            "VATURNOVER_mean":VATURNOVER_mean,"VATURNOVER_std":VATURNOVER_std,"TURNOVER_mean":TURNOVER_mean,"TURNOVER_std":TURNOVER_std}
        
        print("%s;%s;%s"%(stock_no,str(last_trade_date),json.dumps(ret))) 


def static_seq(bykey,sql):
    df = pd.read_sql(sql, conn)
    
    li_ = []
    for idx,row in df.iterrows():
        data_json = json.loads(row['data_json'])
        ret = [data_json.get(field) for field in forecast_fields]
        # print(ret) 
        li_.append(ret)
    
    df_v = pd.DataFrame(li_,columns=forecast_fields)
    df_v_d = df_v.describe() 
    # print(df_v_d)
    
    static_li = [bykey,int(df_v_d['f_high_mean_rate']["count"])]
    for f1 in forecast_fields:
        for f2 in static_fields:
            static_li.append(round(df_v_d[f1][f2],4))
            
    # print(";".join( [str(x) for x in static_li])) 
    
    return static_li

def static_forecast_val():
    # 按天/股票统计最大值，最小值的均值？
    t_li = []
    dates = load_trade_dates(conn)
    for trade_date in dates:
        sql = "select * from stock_for_transfomer where trade_date=%s" % (trade_date) 
        t_li.append(static_seq(trade_date,sql)) 
    df_v = pd.DataFrame(t_li,columns=get_all_fields())
    df_v.to_csv("data/static_seq_dates.txt",sep=";",index=False)  
    
    # stock_no='002913' 
    
    s_li = []
    stocks = load_stocks(conn)
    for stock in stocks:
        stock_no = stock[0]
        sql = "select * from stock_for_transfomer where stock_no='%s'" % (stock_no) 
        try:
            s_li.append(static_seq(stock_no,sql))  
        except:
            print("error:", stock_no)
    df_v = pd.DataFrame(s_li,columns=get_all_fields())
    df_v.to_csv("data/static_seq_stocks.txt",sep=";",index=False)

# python statistics.py > stocks_statistics.jsonl
if __name__ == "__main__":
    # python statistics.py > per_stocks_mean_std.jsonl
    # compute_per_stocks_mean_std()
    
    static_forecast_val()
    # print(get_all_fields())
    
    # compute_mean_std("002913") # 601998,  002913
    
    # compute_mean_std()
    # get_update_sql()