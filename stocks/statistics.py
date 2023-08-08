#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import time
import sqlite3  
import pandas as pd
import json
from common import load_stocks

MAX_ROWS_COUNT = 2000 #从数据库中加载多少数据, 差不多8年的交易日数。

conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)


def c_round(x):
    return round(x,4)

stocks = load_stocks()
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

# python statistics.py > stocks_statistics.jsonl

# import to sqlite3
    
# if __name__ == "__main__":
#     pass 