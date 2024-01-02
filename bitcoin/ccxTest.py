#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os.path  # To manage paths
import itertools
import datetime
import time 
import pandas as pd
import ccxt

# print (ccxt.exchanges) #支持的交易所列表

print(dir(ccxt.kraken()))   

# for exchange_id in ccxt.exchanges:
#     # print(exchange_id)
#     exchange_class = getattr(ccxt, exchange_id)
#     exchange = exchange_class({})
    # print(exchange_id,exchange.get_sandbox_mode())

exchange = ccxt.okcoin () # default id
# okcoin1 = ccxt.okcoin ({ 'id': 'okcoin1' })
# okcoin2 = ccxt.okcoin ({ 'id': 'okcoin2' })
# id = 'btcchina'
# btcchina = eval ('ccxt.%s ()' % id)
# coinbasepro = getattr (ccxt, 'coinbasepro') ()

delay = 2 # seconds
for symbol in exchange.markets:
    print (exchange.fetch_order_book(symbol))
    time.sleep (delay) # rate limit

if exchange.has['fetchOHLCV']:
    for symbol in exchange.markets:
        time.sleep (exchange.rateLimit / 1000) # time.sleep wants seconds
        print (symbol, exchange.fetch_ohlcv (symbol, '1d')) # one day

limit = 10
exchange.fetch_order_book('BTC/USD', limit)

# from variable id
# exchange_id = 'binance' #binance
# exchange_class = getattr(ccxt, exchange_id)
# exchange = exchange_class({
#     'apiKey': 'YOUR_API_KEY',
#     'secret': 'YOUR_SECRET',
# })

# markets = exchange.load_markets()

# exchange.set_sandbox_mode(True)