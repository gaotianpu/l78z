#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os.path  # To manage paths
import datetime 
import backtrader as bt
# from __future__ import absolute_import, division, print_function,unicode_literals

def prepare_data():
    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, '../../datas/orcl-1995-2014.txt')
    print(datapath)
    #5037,Date,Open,High,Low,Close,Adj Close,Volume
    datapath = '/mnt/d/github/backtrader/datas/orcl-1995-2014.txt' 

    # Create a Data Feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        # Do not pass values before this date
        fromdate=datetime.datetime(2000, 1, 1),
        # Do not pass values after this date
        todate=datetime.datetime(2000, 12, 31),
        reverse=False)
    # print(data)
    return data

# Create a Stratey
class TestStrategy(bt.Strategy):
    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])
        
        if self.dataclose[0] < self.dataclose[-1]:
            # current close less than previous close

            if self.dataclose[-1] < self.dataclose[-2]:
                # previous close less than the previous close

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.buy() #价格，挂单？


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    initCrash = 100000.0
    cerebro.broker.setcash(initCrash) #Setting the Cash
    
    # Add a strategy
    cerebro.addstrategy(TestStrategy)
    
    data = prepare_data()
    cerebro.adddata(data) #支持的数据格式？
    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.run()

    finalCrash = cerebro.broker.getvalue()
    pct = (finalCrash-initCrash)*100/initCrash
    print(f'Final Portfolio Value:  {finalCrash: .2f}, {pct:.2f}%')