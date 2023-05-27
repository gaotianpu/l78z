#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
from tkinter.font import names
import pandas as pd
from download_daily import get_stock_prices

def load_prices():
    d = {}
    with open("data/today_price.txt",'r') as f:
        for i,line in enumerate(f):
            parts = line.strip().split(",")
            stock = parts[0]
            d[stock] = (float(parts[7]) - float(parts[2]))/(float(parts[2])+0.0000001)
    return d 

def format_stocks(stocks):
    for stock in stocks:
        yield "%s%s" % ("sh" if stock.startswith("6") else "sz", stock)

def load_current_prices(stocks):
    stocks_f = format_stocks(stocks)
    stock_prices = get_stock_prices(list(stocks_f)) 
    # 0 603656 # stock
    # 1 14.0   # open
    # 2 14.78  # last_close
    # 3 13.58  # current  #当前价？
    # 4 14.29  # high
    # 5 13.4   # low 
    # 6 13.58  # ? 收盘？
    # 7 13.59  # ? 收盘?
    # 8 18143838.0
    # 9 249310939.0
    # 10 76140.0 

    df = pd.DataFrame(stock_prices,
        columns="STOCK,open,last,current,high,low,unk1,unk2,unk3,unk4,unk5".split(","))
    
    df.insert(1, "open_last_rate", 0.0)
    df.insert(1, "current_open_rate", 0.0)
    df.insert(1, "current_last_rate", 0.0)
    df['open_last_rate'] = round((df['open'] - df['last']) * 100 / (df['last'] - 0.00000001),1)
    df['current_open_rate'] = round((df['current'] - df['open']) * 100 / (df['open'] - 0.00000001),1) 
    df['current_last_rate'] = round((df['current'] - df['last']) * 100 / (df['last'] - 0.00000001),1) 
    df = df.sort_values(by="open_last_rate", ascending=False)
    # print(df)

    # 取高开部分 open >= last_close
    # 或者 current > last_close,  > open
    # print("\n".join([",".join([str(x) for x in p]) for p in stock_prices]))
    return df



def run():
    # prices = load_prices()
    df_predict_scores = pd.read_csv("predict/predict_scores_final_today.csv",dtype={'STOCK':str})
    df_predict_scores = df_predict_scores[df_predict_scores.num>0]
    stocks = df_predict_scores["STOCK"].tolist()
    
    current_prices = load_current_prices(stocks)
    # return 

    df_merge = pd.merge(current_prices, df_predict_scores, how='left', on=['STOCK'])

    
    # df_predict_scores.insert(0, "rate", 0.0)
    # for index, row in df_predict_scores.iterrows():
    #     df_predict_scores.loc[index, "rate"] = round(prices.get(row["STOCK"], 0)*100,1)

    df_merge.to_csv("predict/predict_scores_price_%s.csv" % (datetime.datetime.now().strftime('%Y%m%d_%H%M')) , 
        index=False)
    df_merge.to_csv("predict/predict_scores_price.csv", index=False)

run()