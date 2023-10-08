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
from common import load_stocks

PROCESSES_NUM = 5

FUTURE_DAYS = 4 # 预测未来几天的数据, 2,3,5? 2比较合适，3则可能出现重复，再不恰当的数据集划分策略下，训练集和测试可能会重叠？
PAST_DAYS = 20 #使用过去几天的数据做特征

MAX_ROWS_COUNT = 3000 #从数据库中加载多少数据, 差不多8年的交易日数。

# select max(trade_date) from stock_for_transfomer;
MIN_TRADE_DATE = 0 #20230825

conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)

mean_std = {"OPEN_price_mean": 17.9043, "OPEN_price_std": 31.569, "CLOSE_price_mean": 17.9258, "CLOSE_price_std": 31.5798, "change_amount_mean": 0.0076, "change_amount_std": 1.1697, "change_rate_mean": 0.0529, "change_rate_std": 3.4192, "LOW_price_mean": 17.5473, "LOW_price_std": 30.9598, "HIGH_price_mean": 18.3079, "HIGH_price_std": 32.2353, "TURNOVER_mean": 157735.6861, "TURNOVER_std": 333532.135, "TURNOVER_amount_mean": 19274.754, "TURNOVER_amount_std": 42239.1776, "TURNOVER_rate_mean": 2.7719, "TURNOVER_rate_std": 4.2903}

FIELDS = "OPEN_price,CLOSE_price,change_amount,change_rate,LOW_price,HIGH_price,TURNOVER,TURNOVER_amount,TURNOVER_rate".split(",")


log_file = "log/seq_preprocess.log"
logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(lineno)d:%(funcName)s:%(message)s')


def zscore(x,mean,std):
    return round((x-mean)/(std+0.00000001),4)

def compute_rate(x,base): #计算涨跌幅度
    return round((x-base)/base,4)

class PreProcessor:
    def __init__(self, conn,stock_no, future_days = 3,past_days = 20 , data_type="train" , start_trade_date=0):
        self.conn = conn
        self.stock_no = stock_no
        self.future_days = future_days
        self.past_days = past_days
        self.data_type = data_type
        self.start_trade_date = start_trade_date
        self.val_ranges = [-0.00493,0.01434,0.02506,0.03954,0.04997,0.06524,0.09353]
    
    def map_val_range(self, val):
        for i,val_range in enumerate(self.val_ranges):
            if val<val_range:
                return i+1
        return 8 
    
    def process_row(self, df, idx,current_date):
        # current_date = int(df.loc[idx]['trade_date'])
        ret = {"stock_no": self.stock_no, "current_date":current_date}
        
        # 未来值,FUTURE_DAYS最高价，最低价？
        if idx>0: #train
            buy_base = df.loc[idx-1]['OPEN_price'] # df_future.iloc[-1]['TOPEN']
            hold_base = df.loc[idx]['CLOSE_price'] #昨收价格
            
            #用于预测要卖出的价位
            next_high_rate = compute_rate(df.loc[idx-1]['HIGH_price'],hold_base)
            #用于预测要买入的价位
            next_low_rate = compute_rate(df.loc[idx-1]['LOW_price'],hold_base)
            #是否会跌停的判断？还没想清楚
            next_close_rate = compute_rate(df.loc[idx-1]['CLOSE_price'],hold_base)
            
            df_future = df.loc[idx-self.future_days : idx-2] #由于t+1,计算买入后第二个交易的价格
            highest = df_future['HIGH_price'].max()
            lowest = df_future['LOW_price'].min()
            high_mean = df_future['HIGH_price'].mean()
            low_mean = df_future['LOW_price'].mean() 
            
            f_high_rate = compute_rate(highest,buy_base) #round((highest-buy_base)/buy_base,2)
            f_low_rate = compute_rate(lowest,buy_base)  #round((lowest-buy_base)/buy_base,2)
            f_high_mean_rate = compute_rate(high_mean,buy_base) 
            f_low_mean_rate = compute_rate(low_mean,buy_base) 
            
            # f_high_mean_rate
            val_label = self.map_val_range(f_high_mean_rate)
            
            # f_{buy/hold}_{high/low}_{est/mean}_{2(days)}
            # buy, 选股，股票没买入，基线价格=下一个交易日的开盘价
            # hold, 持股，股票已买入，基线价格=当天收盘价
            # high/low,
            # est, highest,lowest, 未来n天内，最高价、最低价的极值
            # mean, 未来n天内，最高价、最低价的均值
            # days, 预测未来几天的，buy情况下，最少2天；hold情况，最少一天；极值按最少的天数计算，均值则可以3,5这样的值？
            
            # f_buy_high_est #为避免拆分时重复导致过拟合，只预测最小天的？
            # f_buy_low_est
            # f_buy_high_mean_3 #预测未来时间上，一次全都生成了？
            # f_buy_low_mean_3
            # f_buy_high_mean_5
            # f_buy_low_mean_5
            
            # f_hold_high_est
            # f_hold_low_est
            # f_hold_high_mean_3
            # f_hold_low_mean_3
            # f_hold_high_mean_5
            # f_hold_low_mean_5
            
            ret['f_high_rate'] = f_high_rate
            ret['f_low_rate'] = f_low_rate
            ret['f_high_mean_rate'] = f_high_mean_rate 
            ret['f_low_mean_rate'] = f_low_mean_rate 
            ret['next_high_rate'] = next_high_rate 
            ret['next_low_rate'] = next_low_rate 
            ret['next_close_rate'] = next_close_rate 
            ret['val_label'] = val_label 
        else: #predict
            ret['f_high_rate'] = 0.0
            ret['f_low_rate'] = 0.0
            ret['f_high_mean_rate'] = 0.0
            ret['f_low_mean_rate'] = 0.0 
            ret['next_high_rate'] = 0.0 
            ret['next_low_rate'] = 0.0 
            ret['next_close_rate'] = 0.0
            ret['val_label'] = 0
        
        # 获取过去所有交易日的均值，标准差
        # 只能是过去的，不能看到未来数据？ 还是说应该固定住？似乎应该固定住更好些 
        
        # 特征值归一化
        ret["past_days"] = []
        df_past = df.loc[idx:idx+self.past_days-1]
        for _,row in df_past.iterrows(): 
            feather_ret = []
            for field in FIELDS:
                feather_ret.append(zscore(row[field],mean_std.get(field+"_mean"),mean_std.get(field+"_std")))
            ret["past_days"].insert(0,feather_ret) 
            
        # 额外;分割的datestock_uid,current_date,stock_no,dataset_type, 便于后续数据集拆分、pair构造等
        datestock_uid = str(ret["current_date"]) + ret['stock_no'] 
        # if len(ret["past_days"]) == self.past_days:
        #     print("%s;%s;%s;0;%s" % (datestock_uid,ret["current_date"],ret['stock_no'],
        #                                 json.dumps(ret))) #
        # #new, 用于list排序
        if len(ret["past_days"]) == self.past_days:
            print("%s;%s;%s;0;%s;%s" % (datestock_uid,ret["current_date"],ret['stock_no'],
                                    ret['val_label'],json.dumps(ret))) #
        

    def process_train_data(self):
        sql = "select * from stock_raw_daily_2 where stock_no='%s' and OPEN_price>0 order by trade_date desc limit 0,%d"%(self.stock_no,MAX_ROWS_COUNT)
        df = pd.read_sql(sql, self.conn) 
        
        end_idx = len(df) - self.past_days + 1
        for idx in range(self.future_days, end_idx):
            try:
                # 保证是增量生成
                current_date = int(df.loc[idx]['trade_date'])
                if current_date <= self.start_trade_date:
                    break
                 
                self.process_row(df, idx,current_date)
            except:
                logging.warning("process_train_data process_row error stock_no=%s,idx=%s" %(self.stock_no,idx) )
     
            # break #debug
            
        df = None  # 释放内存？
        del df  
        
    def process_predict_data(self):
        # statistics = self.get_statistics() 
         
        # max_trade_date = self.get_max_trade_date() #
        sql = "select * from stock_raw_daily_2 where stock_no='%s' and OPEN_price>0 order by trade_date desc limit 0,%d"%(self.stock_no,self.past_days+self.future_days)
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
    stocks = load_stocks(conn)
    time_start = time.time()
    for i, stock in enumerate(stocks):
        p = PreProcessor(conn,stock[0],FUTURE_DAYS,PAST_DAYS,data_type,MIN_TRADE_DATE)
        if processes_idx < 0 or data_type!="train": #predict耗时少，不用拆分
            p.process() 
        elif i % PROCESSES_NUM == processes_idx:
            p.process()
        
    time_end = time.time() 
    time_c = time_end - time_start   #运行所花时间
    # print('time-cost:', time_c, 's') #predict=110s, train=157*400/60=17.5 hours ?
    
    # 解决生成速度慢的方案
    # 1. 并行
    # 2. 增量添加
    conn.close()

def process_new_stocks(data_type):
    with open('uncollect_stock_no.txt','r') as f:
        for line in f:
            fields = line.strip().split(',') 
            stock_no = fields[0]
            p = PreProcessor(conn,stock_no,FUTURE_DAYS,PAST_DAYS,data_type,MIN_TRADE_DATE)
            p.process() 
            
# python seq_preprocess.py train 0 > data/seq_train.txt_0 &
# python seq_preprocess.py predict > data/rnn_predict.txt &
if __name__ == "__main__":
    data_type = sys.argv[1]
    process_idx = -1 if len(sys.argv) != 3 else int(sys.argv[2])
    process_all_stocks(data_type, process_idx)
    
    # process_new_stocks(data_type)
    
    # p = PreProcessor(conn,"000001",FUTURE_DAYS,PAST_DAYS, data_type)
    # p.process()
