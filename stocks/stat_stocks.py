#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
# import numpy as np
import pandas as pd
from common import load_stocks

CN_NAMES = "日期,股票代码,收盘价,最高价,最低价,开盘价,前收盘,涨跌额,涨跌幅,换手率,成交量,成交金额,总市值,流通市值".split(
    ",")
NAMES = "DATE;STOCK;TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP".split(
    ";")

def statis_info(df):
    desc = df.describe()  
    # print(desc)
    # 最大，最小，均值，1/4, 中位数， 3/4
    li = [desc.loc['max']['HIGH'],desc.loc['min']['LOW'],desc.loc['mean']['TCLOSE'],desc.loc['25%']['TCLOSE'],desc.loc['50%']['TCLOSE'],desc.loc['75%']['TCLOSE'],desc.loc['std']['TCLOSE']]
    return  [str(int(desc.loc['count']['HIGH']))+","+",".join([("%0.2f"%(item)) for item in li])]
    # return li
    

def process_stock(stock_no):
    data_file = "data/history/%s.csv" % (stock_no)
    if not os.path.isfile(data_file):
        return

    #1. 指定浮点数的精度，小数点后2位即可？
    #2. 股票派息后，下载到的数据不一致？
    df = pd.read_csv(data_file,
                     names=NAMES, dtype={'STOCK': str})
    
    df = df[df['TCLOSE']>0]
    info_li = [stock_no]

    # 1 年周期
    info_li = info_li + statis_info(df.head(250))

    # 3年周期
    info_li = info_li + statis_info(df.head(730))

    # 7年周期
    info_li = info_li + statis_info(df.head(1700))

    # 全部 
    info_li = info_li + statis_info(df)
    
    print(",".join(info_li))



def test():
    # process_stock('000725')

    stocks = load_stocks()
    print("stockno,count_250,high_250,low_250,mean_250,s25_250,s50t_250,s75_250,std_250,count_730,high_730,low_730,mean_730,s25_730,s50_730,s75_730,std_730,count_1700,high_1700,low_1700,mean_1700,s25_1700,s50_1700,s75_1700,std_1700,count_all,high_all,low_all,mean_all,s25_all,s50_all,s75_all,std_all")
    for i,stock in enumerate(stocks):
        process_stock(stock[0])
        # if i >5:
        #     break


if __name__ == "__main__":
    test()
