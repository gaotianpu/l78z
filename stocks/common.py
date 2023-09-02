#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 60:
# http://www.sse.com.cn/assortment/stock/list/share/
# 

def load_stocks():
    with open("schema/stocks.txt", 'r') as f:
        l = []
        for i, line in enumerate(f):
            fields = line.strip().split(',')
            # 0:stock_no,1:start_date,2:stock_name,3:是否退市
            if fields[3] == "1":
                continue
            yield fields

if __name__ == "__main__":
    stocks = load_stocks()
    for stock in stocks:
        print(stock)