#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import time
import sqlite3  
import json
import logging
import pandas as pd
from common import load_stocks,c_round

PROCESSES_NUM = 5

FUTURE_DAYS = 5 # 预测未来几天的数据, 2,3,5? 2比较合适，3则可能出现重复，在不恰当的数据集划分策略下，训练集和测试可能会重叠？
PAST_DAYS = 20 #使用过去几天的数据做特征

#MAX_ROWS_COUNT = 3000 #从数据库中加载多少数据, 差不多8年的交易日数。

# select max(trade_date) from stock_for_transfomer;
MIN_TRADE_DATE = 0 #20230825

conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)

FIELDS_PRICE="open_price;close_price;low_price;high_price".split(";")
FIELDS = "change_amount;turnover;turnover_amount;turnover_rate".split(";")

#low_low,low_high,high_low,high_high #昨日的最低最高价和今日的最高最低价对比
FIELDS_RATE="open_rate;change_rate;low_rate;high_rate;high_low_range;open_low_rate;open_high_rate;open_close_rate".split(";")
FIELDS_IDX="turnover_idx;turnover_amount_idx;turnover_rate_idx;change_rate_idx;last_close_idx;open_rate_idx;low_rate_idx;high_rate_idx;high_low_range_idx;open_low_rate_idx;open_high_rate_idx;open_close_rate_idx".split(";")


log_file = "log/seq_preprocess.log"
logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(lineno)d:%(funcName)s:%(message)s')

def minmax(x,min,max):
    val = round((x-min)/(max-min),4)
    if val>1:
        val = 1
    if val<0:
        val = 0

def zscore(x,mean,std):
    return round((x-mean)/(std+0.00000001),4)

def compute_rate(x,base): #计算涨跌幅度
    return round((x-base)/base,4)

# def get_stocks_static():
#     stocks_statics = {} 
#     df = pd.read_sql("select stock_no,data_json from stock_statistics_info", conn)
#     for idx,row in df.iterrows():
#         stocks_statics[row['stock_no']] = json.loads(row['data_json'])
#     return stocks_statics
            
class PreProcessor:
    def __init__(self, conn,stock_no, future_days = 2,past_days = 20 , data_type="train", start_trade_date=0):
        self.conn = conn
        self.stock_no = stock_no
        self.future_days = future_days
        self.past_days = past_days
        self.data_type = data_type
        self.start_trade_date = start_trade_date
        self.val_ranges = [-1.400000e-03,1.380000e-02,2.280000e-02,3.540000e-02,4.470000e-02,5.840000e-02,0.068900,0.084600,0.110700,0.122000,0.138700,0.169200]
        # self.stock_static = stock_static #stock_static,
        
    def map_val_range(self, val):
        return len([item for item in self.val_ranges if val>item])
            
        # for i,val_range in enumerate(self.val_ranges):
        #     if val<val_range:
        #         return i+1
        # return 8 
    
    def process_row(self, df, idx,current_date):
        # current_date = int(df.loc[idx]['trade_date'])
        ret = {"stock_no": self.stock_no, "current_date":current_date}
        
        (highN_rate,next_high_rate,next_low_rate) = (0.0,0.0,0.0)
        
        # 未来值,FUTURE_DAYS最高价，最低价？
        if idx>0: #train
            buy_base = df.loc[idx-1]['open_price'] # df_future.iloc[-1]['TOPEN']
            highN_rate = compute_rate(df.loc[idx-self.future_days]['high_price'],buy_base) #保守点，把最高价?作为买入价
            
            hold_base = df.loc[idx]['close_price'] #昨收价格 
            #用于预测要买入的价位
            next_low_rate = compute_rate(df.loc[idx-1]['low_price'],hold_base)
            #用于预测要卖出的价位
            next_high_rate = compute_rate(df.loc[idx-1]['high_price'],hold_base)
            
            #是否会跌停的判断？还没想清楚
        
            ret['highN_rate'] = highN_rate
            ret['next_high_rate'] = next_high_rate
            ret['next_low_rate'] = next_low_rate
            ret['val_label'] = 0 
        else: #for predict
            ret['highN_rate'] = highN_rate
            ret['next_high_rate'] = next_high_rate
            ret['next_low_rate'] = next_low_rate
            ret['val_label'] = 0 
        
        # 获取过去所有交易日的均值，标准差
        # 只能是过去的，不能看到未来数据？ 还是说应该固定住？似乎应该固定住更好些 
        
        # 特征值归一化
        ret["past_days"] = []
        
        df_60 = df.loc[idx:idx+60]
        
        df_past = df.loc[idx:idx+self.past_days-1]
        
        max_price = df_60['high_price'].max()
        min_price = df_60['low_price'].min()
        min_max_rate={}
        for rate in FIELDS:
            min_max_rate[rate]=[df_60[rate].min(),df_60[rate].max()]
            
        for _,row in df_past.iterrows(): 
            feather_ret = []
            # 开盘、收盘、最高、最低价，采用过去PAST_DAYS=20天内的最高价、最低价，作为min-max归一化
            for field in FIELDS_PRICE:
                feather_ret.append(minmax(row[field],min_price,max_price))
                
            for field in FIELDS:
                feather_ret.append(minmax(row[field],min_max_rate[field][0],min_max_rate[field][1]))
            
            # 对成交金额，使用全局的minmax归一化？
            # select max(turnover_amount),min(turnover_amount) from stock_raw_daily_2;
            feather_ret.append(minmax(row["turnover_amount"],0.01,4796717.0)) 
        
            # 对成交量，使用全局的minmax归一化？
            # select max(turnover),min(turnover) from stock_raw_daily_2;
            feather_ret.append(minmax(row["turnover"],0.0,41144528.0))
            
            #与前一天对比
            for field in FIELDS_PRICE+FIELDS:
                feather_ret.append((row[field] - df.loc[row.index+1][field])/df.loc[row.index+1][field])
            
            for field in FIELDS_PRICE:
                feather_ret.append((row[field] - row['last_close'])/row['last_close'])
            
            feather_ret.append((row['low_price'] - row['open_price'])/row['open_price'])
            feather_ret.append((row['high_price'] - row['open_price'])/row['open_price'])
            
            
            # for field in FIELDS_RATE: #+FIELDS_IDX:
            #     feather_ret.append(row[field]) 
                
                # mean = self.stock_static.get(field+"_mean")
                # std = self.stock_static.get(field+"_std")
                # # print(field,mean,std)
                # feather_ret.append(zscore(row[field],mean,std))
            ret["past_days"].insert(0,feather_ret) 
            
        # 额外;分割的datestock_uid,current_date,stock_no,dataset_type, 便于后续数据集拆分、pair构造等
        datestock_uid = str(ret["current_date"]) + ret['stock_no']
        if len(ret["past_days"]) == self.past_days:
            li = [str(item) for item in [datestock_uid,ret["current_date"],ret['stock_no'],0,
                                         highN_rate,next_high_rate,next_low_rate,
                                    ret['val_label'],json.dumps(ret)]] #
            print(';'.join(li))
            
        

    def process_train_data(self):
        sql = "select * from stock_raw_daily_1 where stock_no='%s' and open_price>0 order by trade_date desc"%(self.stock_no)
        df = pd.read_sql(sql, self.conn)
        
        end_idx = len(df) - self.past_days + 1
        for idx in range(self.future_days, end_idx):
            try:
                # 保证是增量生成
                current_date = int(df.loc[idx]['trade_date'])
                # print(current_date,self.start_trade_date)
                # if current_date <= self.start_trade_date:
                #     break
                    
                self.process_row(df, idx,current_date)
            except:
                logging.warning("process_train_data process_row error stock_no=%s,idx=%s" %(self.stock_no,idx) )
     
            # break #debug
            
        df = None  # 释放内存？
        del df  
        
    def process_predict_data(self):
        # statistics = self.get_statistics() 
         
        # max_trade_date = self.get_max_trade_date() #
        sql = "select * from stock_raw_daily_1 where stock_no='%s' and open_price>0 order by trade_date desc limit 0,%d"%(self.stock_no,self.past_days+self.future_days)
        df = pd.read_sql(sql, conn) 
        
        current_date = int(df.loc[0]['trade_date'])
        self.process_row(df, 0,current_date)
        
        # try:
        #     self.process_row(df, 0)
        # except:
        #     logging.warning("process_predict_data process_row error stock_no=%s" %(self.stock_no) )
     
        
        df = None  # 释放内存？
        del df  
    
    def process(self):
        if self.data_type == "train":
            self.process_train_data()
        else:
            self.process_predict_data()

  
def process_all_stocks(data_type="train", processes_idx=-1):
    # stocks_statics = get_stocks_static()
    stocks = load_stocks(conn)
    time_start = time.time()
    for i, stock in enumerate(stocks):
        # stock_static = stocks_statics.get(stock[0],None)
        # assert stock_static # ?   #stock_static,
        p = PreProcessor(conn,stock[0],FUTURE_DAYS,PAST_DAYS,data_type,MIN_TRADE_DATE)
        if processes_idx < 0 or data_type!="train": #predict耗时少，不用拆分
            p.process() 
        elif i % PROCESSES_NUM == processes_idx:
            p.process()
        # print(stock[0])
        # break 
        
    time_end = time.time() 
    time_c = time_end - time_start   #运行所花时间
    # print('time-cost:', time_c, 's') #predict=110s, train=157*400/60=17.5 hours ?
    
    # 解决生成速度慢的方案
    # 1. 并行
    # 2. 增量添加
    conn.close()

def process_new_stocks(data_type):
    # stocks_statics = get_stocks_static()
    with open('uncollect_stock_no.txt','r') as f:
        for line in f:
            fields = line.strip().split(',') 
            stock_no = fields[0]
            #stocks_statics[stock_no],
            p = PreProcessor(conn,stock_no,FUTURE_DAYS,PAST_DAYS,data_type,MIN_TRADE_DATE)
            p.process() 

def convert_stock_raw_daily(start_date=None):
    '''基于stock_raw_daily_2，扩展出新的字段'''
    if not start_date:
        sql = "select max(trade_date) as trade_date from stock_raw_daily_1"
        df = pd.read_sql(sql, conn)
        start_date = df['trade_date'][0]
        print(start_date) 

    sql = "select distinct trade_date from stock_raw_daily_2 where trade_date>%s order by trade_date" % (start_date)
    df = pd.read_sql(sql, conn)
    trade_dates = df['trade_date'].sort_values(ascending=False).tolist()
    print("len(trade_dates)=",len(trade_dates))
    
    for idx,date in enumerate(trade_dates):  
        print(idx,date)
        sql = "select * from stock_raw_daily_2 where trade_date='%s' order by stock_no"%(date)
        df_date = pd.read_sql(sql, conn)
        
        cnt = len(df_date)
        
        df_date['change_rate'] = df_date.apply(lambda x: c_round(x['change_rate']/100), axis=1) 
        df_date['last_close'] = df_date['close_price'] - df_date['change_amount'] 
        df_date['open_rate'] = c_round((df_date['open_price'] - df_date['last_close']) / df_date['last_close']) 
        df_date['low_rate'] = c_round((df_date['low_price'] - df_date['last_close']) / df_date['last_close']) 
        df_date['high_rate'] = c_round((df_date['high_price'] - df_date['last_close']) / df_date['last_close']) 
        df_date['high_low_range'] = c_round(df_date['high_rate'] - df_date['low_rate'])
        df_date['open_low_rate'] = c_round((df_date['low_price'] - df_date['open_price']) / df_date['open_price']) 
        df_date['open_high_rate'] = c_round((df_date['high_price'] - df_date['open_price']) / df_date['open_price']) 
        df_date['open_close_rate'] = c_round((df_date['close_price'] - df_date['open_price']) / df_date['open_price']) 
        
        # 当天交易中，各种字段相对位置
        fields = 'turnover,turnover_amount,turnover_rate,change_rate,last_close,open_rate,low_rate,high_rate,high_low_range,open_low_rate,open_high_rate,open_close_rate'.split(',')
        for field in fields:
            df_date = df_date.sort_values(by=[field]) # 
            df_date[field+"_idx"] = [c_round((idx+1)/cnt) for idx in range(cnt)]
        
        df_date = df_date.sort_values(by=['stock_no']) # 
        df_date.to_csv("data/trade_dates/%s.txt"%(date),sep=";", header=None,index=False) 
        # df_date.to_csv("tmp_%s.txt"%(date),sep=";", index=False) 
        # break            

if __name__ == "__main__":
    data_type = sys.argv[1]
    if data_type in "train,predict".split(","):
        process_idx = -1 if len(sys.argv) != 3 else int(sys.argv[2])
        process_all_stocks(data_type, process_idx)
    
    if data_type == "convert_stock_raw_daily":
        convert_stock_raw_daily()
    
    # process_new_stocks(data_type)
    
    # p = PreProcessor(conn,"000001",FUTURE_DAYS,PAST_DAYS)
    # p.process()
