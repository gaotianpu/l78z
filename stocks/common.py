#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd 

# 60:
# http://www.sse.com.cn/assortment/stock/list/share/
# 

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
    
    

if __name__ == "__main__":
    conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)
    stocks = load_stocks(conn)
    for stock in stocks:
        print(stock)
        break 
    
    stocks = load_stocks_f()
    for stock in stocks:
        print(stock)
        break 