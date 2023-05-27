#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最新数据下载 - 网易数据源
"""
import os
import sys
import time
import datetime
import math
import requests
import logging
from multiprocessing import Pool
from itertools import islice

PAGE_SIZE = 80
DAILY_DATA_URL = "https://hq.sinajs.cn/list="
#                 https://hq.sinajs.cn/list=sh600000,sz300026,sz002177,sh600006,sz000007


log_file = "log/download_daily.log"
os.system("rm -f " + log_file)
logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(lineno)d:%(funcName)s:%(message)s')


def load_stocks():
    with open("schema/stocks.txt", 'r') as f:
        l = []
        for i, line in enumerate(f):
            fields = line.strip().split(',')
            #0:stock_no,1:start_date,2:stock_name,3:是否退市
            if fields[3] == "1":
                continue
            yield "%s%s" % ("sh" if fields[0].startswith("6") else "sz", fields[0])


def download(url):
    resp = None
    headers={'Referer':'https://finance.sina.com.cn'}
    for i in range(3):  # 失败最多重试3次
        try:
            resp = requests.get(url=url, headers=headers,timeout=0.5)
            logging.info("download success,retry_times:%s,url:%s" %
                         (i, url))
            break
        except:
            if i == 2:  # 超过最大次数
                logging.warning("download fail,retry_times:%s,url:%s" %
                                (i, url))
                return None 
            else:
                continue

    lines = resp.text.split(';\n')
    rows = []
    for idx, line in enumerate(lines):
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        p1s = parts[0].split('="')
        row = []
        row.append(p1s[0].split('_')[-1][2:])  # stock_no
        # row.append(p1s[1])  # stock_cn_name
        date = " ".join(parts[-4:-2])
        # print(date)
        # row = row + [float(p) for p in parts[1:-1]]
        row = row + [float(p) for p in parts[1:11]] 
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
        
        rows.append(row) 

    return rows
        # print(",".join(li))


def get_stock_prices(stocks):
    stocks_count = len(stocks)
    pagecount = int(math.ceil(stocks_count/PAGE_SIZE))

    li = []
    for i in range(0, pagecount+1):
        url_params = ','.join(stocks[i*PAGE_SIZE:(i+1)*PAGE_SIZE])
        url = DAILY_DATA_URL + url_params
        # print(url)
        tmp = download(url)
        if tmp:
            li = li + tmp
        # break 
    return li 

# python download_daily.py > data/today_price.txt & 
if __name__ == "__main__":
    stocks = list(load_stocks())
    # print(stocks[:5])
    prices = get_stock_prices(stocks)
    print("\n".join([",".join([str(x) for x in p]) for p in prices]))
