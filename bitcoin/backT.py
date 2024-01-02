#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os.path  # To manage paths
import itertools
import datetime 
import pandas as pd 
import backtrader as bt
import backtrader.indicators as btind
import backtrader.feeds as btfeeds
# from __future__ import absolute_import, division, print_function,unicode_literals

def prepare_YahooFinanceCSVData():
    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    # github/backtrader/datas/orcl-1995-2014.txt
    # 5037,Date,Open,High,Low,Close,Adj Close,Volume
    datapath = os.path.join(modpath, 'data/orcl-1995-2014.txt')

    # Create a Data Feed bt.feeds
    data = btfeeds.YahooFinanceCSVData(
        dataname=datapath,
        # Do not pass values before this date
        fromdate=datetime.datetime(2000, 11, 1),
        # Do not pass values after this date
        todate=datetime.datetime(2000, 12, 31),
        reverse=False) # 按日期升序排序
    # print(data)
    return data


def prepare_by_pd():
    '''使用pandas读取数据'''
    dataframe = pd.read_csv('data/btc/all_2014_2023.raw.csv',sep=";",
                    header=0,dtype={'trade_date':str},na_values=0.0)
    dataframe["trade_date"] = pd.to_datetime(dataframe.trade_date)
    dataframe = dataframe.sort_values(by=['trade_date'],ascending=True) #注意按日期升序排序
    data = btfeeds.PandasData(
        dataname=dataframe, datetime=0, open=2, high=3, low=4, close=5, volume=6,
        fromdate=datetime.datetime(2023, 1, 1), todate=datetime.datetime(2023, 12, 15)
    )
    return data


def prepare_csv():
    '''基于GenericCSVData，日期处理仍然存在问题，基于日期的from/to有问题，基于日期的reverse
    '''
    data = btfeeds.GenericCSVData(
        dataname='data/btc/all_2014_2023.raw.csv',
        separator=";", dtformat="%Y%m%d", nullvalue=0.0,
        datetime=0, open=2, high=3, low=4, close=5, volume=6, 
        openinterest=-1, #未平仓合约
        reverse=False,
        
        # fromdate=datetime.datetime(2023, 1, 1), 
        # todate=datetime.datetime(2023, 12, 16), #有问题，全被过滤了？ 
        
        # tmformat=('%Y%m%d'),  
    )
    return data 

    
class MyCSVData(bt.CSVDataBase):

    # def start(self):
    #     # Nothing to do for this data feed type
    #     pass

    # def stop(self):
    #     # Nothing to do for this data feed type
    #     pass

    def _loadline(self, linetokens):
        print(linetokens)
        if not linetokens:
            return False
        
        i = itertools.count(0)

        dttxt = linetokens[next(i)]
        # Format is YYYY-MM-DD
        y = int(dttxt[0:4])
        m = int(dttxt[4:6])
        d = int(dttxt[6:8])

        dt = datetime.datetime(y, m, d)
        dtnum = int(dttxt) #date2num(dt)
        
        btc_no = linetokens[next(i)]
        print(dt,dtnum)

        self.lines.datetime[0] = dtnum
        self.lines.open[0] = float(linetokens[next(i)])
        self.lines.high[0] = float(linetokens[next(i)])
        self.lines.low[0] = float(linetokens[next(i)])
        self.lines.close[0] = float(linetokens[next(i)])
        self.lines.volume[0] = float(linetokens[next(i)])
        self.lines.openinterest[0] = 0.0  #float(linetokens[next(i)])

        return True


# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
        ('exitbars', 5),
        ('printlog', False),
    )
    
    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        
        # To keep track of pending orders
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # Add a MovingAverageSimple indicator
        self.sma = btind.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function for this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log( 'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price, order.executed.value, order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                     (order.executed.price, order.executed.value, order.executed.comm))

            self.bar_executed = len(self) # len(self) 是啥意思？
            # self.log(f'bar_executed:{self.bar_executed}') 记录订单完成时，当前执行到多少条了。
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))
            
    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.dataclose[0])
        
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return 
        
        # Check if we are in the market
        # self.log(self.position)
        # Size, Price, Price orig, Adjbase,Closed,Opened
        if not self.position: #？？只能全部卖出之后，再买入?
            if self.dataclose[0] > self.sma[0]:
                
            # if self.dataclose[0] < self.dataclose[-1]:
            #     # current close less than previous close

            #     if self.dataclose[-1] < self.dataclose[-2]:
                    # previous close less than the previous close

                    # BUY, BUY, BUY!!! (with all possible default parameters)
                    self.log('BUY CREATE, %.2f' % self.dataclose[0]) 
                    # Keep track of the created order to avoid a 2nd order
                    self.order = self.buy() #成交价格，挂单数量？
        else:
            if self.dataclose[0] < self.sma[0]:
            # Already in the market ... we might sell
            # if len(self) >= (self.bar_executed + self.params.exitbars): #有点疑问
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                # self.log(f'len={len(self)},bar_executed={self.bar_executed} self.position={self.position}')
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()
    
    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f, cash=%.2f' %
                 (self.params.maperiod, self.broker.getvalue(),self.broker.getcash()), doprint=True)


def run():
    # 0. 初始化，账户金额，佣金
    cerebro = bt.Cerebro()
    
    initCrash = 10000000.0
    cerebro.broker.setcash(initCrash) #Setting the Cash
    cerebro.broker.setcommission(commission=0.0001) #佣金
    
    # Add a FixedSize sizer according to the stake
    # https://www.backtrader.com/docu/sizers-reference/
    # FixedSize, stake | tranches
    # FixedReverser, 
    # PercentSizer, PercentSizerInt
    # AllInSizer, AllInSizerInt
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    
    # 1. 准备数据
    # data = prepare_YahooFinanceCSVData() # 官方示例
    # data = prepare_csv() # bitcoin csv形式，日期过滤还有问题
    # data = MyCSVData(dataname='data/btc/all_2014_2023.raw.csv',separator=";") #最后一列结束异常？
    data = prepare_by_pd() # bitcoin pandas形式
    cerebro.adddata(data) #支持的数据格式？
    
    # 2. 添加策略，注意：是类，非实例化的对象
    cerebro.addstrategy(TestStrategy,exitbars=5)
    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # 3. 执行
    cerebro.run()
    
    # 4. 效果分析
    finalCrash = cerebro.broker.getvalue()
    pct = (finalCrash-initCrash)*100/initCrash
    print(f'Final Portfolio Value:  {finalCrash: .2f}, {pct:.2f}%')

if __name__ == '__main__':
    run()