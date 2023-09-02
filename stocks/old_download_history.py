#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
历史数据下载 - 网易数据源
"""
import os
import sys
import time
import datetime
import requests
import logging
from multiprocessing import Pool
from itertools import islice
from common import load_stocks

MAX_DAYS = 7  #最多下载多少天的交易数据
PROCESSES_NUM = 3

# https://q.stock.sohu.com/cn/600519/lshq.shtml
# https://q.stock.sohu.com/hisHq?code=cn_600519&start=20230508&end=20230829&stat=1&order=D&period=d&callback=historySearchHandler&rt=jsonp&r=0.15448371743973954&0.6702222141861589

# http://quotes.money.163.com/service/chddata.html?code=0600000&start=20200527&end=20211010&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP
DOWNLOAD_FIELDS = "TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP"
HISTORY_DATA_URL = 'http://quotes.money.163.com/service/chddata.html?code={code}&start={start}&end={end}&fields={fields}'

log_file = "log/download_history.log"
logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(lineno)d:%(funcName)s:%(message)s')

def get_cache_file(stock_no):
    return 'data/history/%s.csv' % (stock_no)


def download(stock_no, start=None, allow_cache=True):
    cache_file = get_cache_file(stock_no)

    if os.path.exists(cache_file):
        if allow_cache:
            return
        else:
            os.system("rm -f " + cache_file)

    if not start:
        start = (datetime.datetime.now() -
                 datetime.timedelta(MAX_DAYS)).strftime("%Y%m%d")
    params = {"code": '0' + stock_no if stock_no.startswith('6') else '1' + stock_no,
              "start": start,
              "end": (datetime.datetime.now()+datetime.timedelta(1)).strftime("%Y%m%d"),
              "fields": DOWNLOAD_FIELDS}
    source_url = HISTORY_DATA_URL.format(**params)

    resp = None
    for i in range(3):  # 失败最多重试3次
        try:
            resp = requests.get(url=source_url, timeout=0.5)
            logging.info("download success,retry_times=%s,stock=%s,url=%s" %
                         (i, stock_no, source_url))
            break
        except:
            if i == 2:  # 超过最大次数
                logging.warning("download fail,retry_times=%s,stock=%s,url=%s" %
                                (i, stock_no, source_url))
                return
            else:
                continue

    # 对返回结果再加工下,方便后续导入db，生成训练数据等
    new_lines = []
    lines = resp.text.split('\n')
    for lineNO, line in enumerate(lines):
        if lineNO == 0:
            continue
        line = line.replace('None', '0')
        fields = line.strip().split(',')
        if len(fields) < 2:
            break
        new_fields = []
        new_fields.append(fields[0].replace('-', ''))
        new_fields.append(fields[1].replace("'", ''))
        new_fields = new_fields + fields[3:]
        new_lines.append(",".join(new_fields))

    with open(cache_file, 'w+') as f:
        f.write('\n'.join(new_lines))
        f.write('\n')


def download_all(processes_idx=-1):
    # 多线程下载？ 线程太多封禁？
    for i, stock in enumerate(load_stocks()):
        if processes_idx < 0 or i % PROCESSES_NUM == processes_idx:
            # download(stock[0],stock[1])
            download(stock[0])


def download_failed(stocknos):
    """加载部分失败的
    # cat log/download_history.log | grep retry_times=2 | sed 's/.*stock=\([0-9]*\).*/\1/g'
    """
    stocks = stocknos.split(",")
    for i, stockno in enumerate(stocks):
        download(stockno)
        time.sleep(0.1)  # 0.1s 间隔


if __name__ == "__main__":
    # download('000725','20010112')
    process_idx = -1 if len(sys.argv) != 2 else int(sys.argv[1])
    download_all(process_idx)

    # download("515790")
