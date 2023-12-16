#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import datetime # import datetime
import sqlite3  
import pandas as pd
from sqlalchemy import true

trade_date = '20220410'
conn = sqlite3.connect("file:newdb/stocks.db?mode=ro", uri=True)

# def get_std(trade_date):
#     sql = "select stock_no,trade_date,TCLOSE,VOTURNOVER from stock_raw_daily where trade_date>'%s' and HIGH>0;"%(trade_date)
#     df = pd.read_sql(sql, conn)
#     group = df.groupby('stock_no')[['TCLOSE','VOTURNOVER']].std()
#     # 需要将TCLOSE,VOTURNOVER 标准差给打平归一化？
#     return group 

# 成交金额 VOTURNOVER

def get_std(trade_date,stock_no,high_price,low_price,max_VOTURNOVER,min_VOTURNOVER):
    max_min_price = high_price - low_price
    max_min_VOTURNOVER = max_VOTURNOVER - min_VOTURNOVER
    sql = f"""select trade_date,(TCLOSE-{low_price})/{max_min_price} as mm_TCLOSE,
        (VOTURNOVER-{min_VOTURNOVER})/{max_min_VOTURNOVER} as mm_VOTURNOVER
        from stock_raw_daily where trade_date>'{trade_date}' and stock_no='{stock_no}'
        and HIGH>0;"""
    # print(sql)
     
    df = pd.read_sql(sql, conn)
    ret = df[['mm_TCLOSE','mm_VOTURNOVER']].std()
    return round(ret['mm_TCLOSE']*100,1),round(ret['mm_VOTURNOVER']*100,1)


def run(trade_date):
    sql="select stock_no,max(HIGH) as high_price,min(LOW) as low_price,max(VOTURNOVER) as max_VOTURNOVER,min(VOTURNOVER) as min_VOTURNOVER from stock_raw_daily where trade_date>'%s' and HIGH>0 group by stock_no;" % (trade_date)
    df = pd.read_sql(sql, conn)

    df.insert(1, 'std_VOTURNOVER', 0.0)
    df.insert(1, "std_TCLOSE", 0.0)
    for index, row in df.iterrows():
        std_TCLOSE,std_VOTURNOVER = get_std(trade_date,
            row["stock_no"],
            row["high_price"] ,
            row["low_price"],
            row["max_VOTURNOVER"] ,
            row["min_VOTURNOVER"])
        # print(row["stock_no"])
        df.loc[index, "std_TCLOSE"] = std_TCLOSE
        df.loc[index, "std_VOTURNOVER"] = std_VOTURNOVER
        # if index>1:
        #     break 
    
    # return 

    dfp = pd.read_csv("data/today_price.txt",header=None,dtype={'stock_no':str},
            names="stock_no,open,last,current,high,low,unk1,unk2,cVOTURNOVER,cTURNOVER,cVATURNOVER".split(","))
    dfp = dfp[dfp['open']>0]
    df_merge = pd.merge(dfp, df, how='left', on=['stock_no'])

    # group = get_std(trade_date)  
    # df_merge = pd.merge(df_merge, group, left_on='stock_no', right_index=True)

    df_merge.insert(1, "mm_VOTURNOVER", 0.0)
    df_merge['mm_VOTURNOVER'] = round((df_merge['cVOTURNOVER'] - df_merge['min_VOTURNOVER']) * 100 / (df_merge['max_VOTURNOVER'] - df_merge['min_VOTURNOVER'] + 0.00000001),1)

    df_merge.insert(1, "mm_price", 0.0)
    df_merge['mm_price'] = round((df_merge['current'] - df_merge['low_price']) * 100 / (df_merge['high_price'] - df_merge['low_price'] + 0.00000001),1)
 
    # 价格，成交量；当前值，最大，最小值得出归一化后的值；历史值的方差判断波动情况
    df_merge.to_csv("predict/maxmin_%s.csv" % (datetime.datetime.now().strftime('%Y%m%d_%H%M')) , 
            index=False)
    df_merge.to_csv("predict/maxmin_today.csv", index=False)

if __name__ == "__main__":
    run(trade_date)
    # get_std(trade_date)
    conn.close()
