#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd 

# 60:
# http://www.sse.com.cn/assortment/stock/list/share/
# 
def c_round(x):
    return round(x,4)

def load_stocks_f():
    with open("schema/stocks.txt", 'r') as f:
        for i, line in enumerate(f):
            fields = line.strip().split(',')
            # 0:stock_no,1:start_date,2:stock_name,3:是否退市
            if fields[3] == "1":
                continue
            yield fields
    
def load_stocks(conn=None):
    sql = "select stock_no,start_date,stock_name,is_drop from stock_basic_info where is_drop<>1"
    df = pd.read_sql(sql, conn)
    # print(df)
    return df.values.tolist()

def load_trade_dates(conn,start_date=20230825):
    sql = "select distinct trade_date from stock_for_transfomer where trade_date>%s"%(start_date)
    df = pd.read_sql(sql, conn)
    return df['trade_date'].sort_values(ascending=False).tolist()

def get_max_trade_date(conn,stock_no,table_name="stock_raw_daily_2"):
    sql = "select max(trade_date) as max_trade_date from %s where stock_no='%s';" %(table_name,stock_no)
    df = pd.read_sql(sql, conn)
    # if len(df):
    return df['max_trade_date'][0]
    # else:
    #     return 0
        # sql = "select min(trade_date) as min_trade_date from %s ;" %(table_name)
        # df = pd.read_sql(sql, conn)
        # return df['min_trade_date'][0]

def load_prices(conn,trade_date):
    sql = "select  cast(trade_date||stock_no as int64) as pk_date_stock,stock_no,CLOSE_price,LOW_price,HIGH_price from stock_raw_daily_2 where trade_date=%s"%(trade_date)
    df = pd.read_sql(sql, conn)
    return df 
    
if __name__ == "__main__":
    conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)
    # ret = get_max_trade_date(conn,stock_no="002914")
    ret = load_prices(conn,20230908)
    print(ret)
    # stocks = load_stocks(conn)
    # for stock in stocks:
    #     print(stock)
    #     break 
    
    # stocks = load_stocks_f()
    # for stock in stocks:
    #     print(stock)
    #     break 