#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import time
import sqlite3  
import pandas as pd
from common import load_stocks

PROCESSES_NUM = 5

FUTURE_DAYS = 3 # 预测未来几天的数据, 2,3,5? 2比较合适，3则可能出现重复，再不恰当的数据集划分策略下，训练集和测试可能会重叠？
PAST_DAYS = 20 #使用过去几天的数据做特征

MAX_ROWS_COUNT = 1000 #从数据库中加载多少数据

max_high,min_high,max_low,min_low = 0.818,-0.6,0.668,-0.6

conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)

def zscore(x,mean,std):
    return round((x-mean)/(std+0.00000001),4)

def compute_rate(x,base): #计算涨跌幅度
    return round((x-base)/base,4)

class PreProcessor:
    def __init__(self, conn,stock_no, future_days = 3,past_days = 20 , data_type="train"):
        self.conn = conn
        self.stock_no = stock_no
        self.future_days = future_days
        self.past_days = past_days
        self.data_type = data_type
    
    def process_row(self, df, idx):
        ret = {"stock_no": self.stock_no, "current_date":df.loc[idx]['trade_date']}
        
        # 未来值,FUTURE_DAYS最高价，最低价？
        if idx>0: #train
            buy_base = df.loc[idx-1]['TOPEN'] # df_future.iloc[-1]['TOPEN']
            hold_base = df.loc[idx]['TCLOSE']
            
            df_future = df.loc[idx-self.future_days : idx-2] #由于t+1,计算买入后第二个交易的价格
            highest = df_future['HIGH'].max()
            lowest = df_future['LOW'].min()
            mean = df_future['HIGH'].mean()
            
            # print(df_future.describe()) 
            # print(base,highest,lowest,mean)
            
            f_high_rate = compute_rate(highest,buy_base) #round((highest-buy_base)/buy_base,2)
            f_low_rate = compute_rate(lowest,buy_base)  #round((lowest-buy_base)/buy_base,2)
            f_mean_rate = compute_rate(mean,buy_base) #round((mean-buy_base)/buy_base,2)
            
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
            ret['f_mean_rate'] = f_mean_rate 
        else: #predict
            ret['f_high_rate'] = 0.0
            ret['f_low_rate'] = 0.0
            ret['f_mean_rate'] = 0.0 
        
        # 获取过去所有交易日的均值，标准差
        # 只能是过去的，不能看到未来数据？ 还是说应该固定住？
        df_history = df.loc[idx: ] 
        df_history_describe = df_history.describe() 
        # 价格
        price_mean = round(df_history_describe["TOPEN"]["mean"],3)
        price_std = round(df_history_describe["TOPEN"]["std"],3) 
        # VOTURNOVER 成交金额
        VOTURNOVER_mean = round(df_history_describe["VOTURNOVER"]["mean"],3)
        VOTURNOVER_std = round(df_history_describe["VOTURNOVER"]["std"],3)
        # VATURNOVER 成交量
        VATURNOVER_mean = round(df_history_describe["VATURNOVER"]["mean"],3)
        VATURNOVER_std = round(df_history_describe["VATURNOVER"]["std"],3)
        #  ('TURNOVER', '换手率')
        TURNOVER_mean = round(df_history_describe["TURNOVER"]["mean"],3)
        TURNOVER_std = round(df_history_describe["TURNOVER"]["std"],3) 
 
        
        # 特征值归一化
        ret["past_days"] = []
        df_past = df.loc[idx:idx+self.past_days-1]
        for _,row in df_past.iterrows():
            # feather_ret = {} 
            # feather_ret["TOPEN"] = zscore(row['TOPEN'],price_mean,price_std)
            # feather_ret["TCLOSE"] = zscore(row['TCLOSE'],price_mean,price_std)
            # feather_ret["HIGH"] = zscore(row['HIGH'],price_mean,price_std)
            # feather_ret["LOW"] = zscore(row['LOW'],price_mean,price_std)
            # feather_ret["LCLOSE"] = zscore(row['LCLOSE'],price_mean,price_std) #有必要么？
            # feather_ret["VOTURNOVER"] = zscore(row['VOTURNOVER'],VOTURNOVER_mean,VOTURNOVER_std)
            # feather_ret["VATURNOVER"] = zscore(row['VATURNOVER'],VATURNOVER_mean,VATURNOVER_std)
            # feather_ret["TURNOVER"] = zscore(row['TURNOVER'],TURNOVER_mean,TURNOVER_std)
            
            feather_ret = []
            feather_ret.append(zscore(row['TOPEN'],price_mean,price_std))
            feather_ret.append(zscore(row['TCLOSE'],price_mean,price_std))
            feather_ret.append(zscore(row['HIGH'],price_mean,price_std))
            feather_ret.append(zscore(row['LOW'],price_mean,price_std))
            feather_ret.append(zscore(row['LCLOSE'],price_mean,price_std)) #有必要么？
            feather_ret.append(zscore(row['VOTURNOVER'],VOTURNOVER_mean,VOTURNOVER_std))
            feather_ret.append(zscore(row['VATURNOVER'],VATURNOVER_mean,VATURNOVER_std))
            feather_ret.append(zscore(row['TURNOVER'],TURNOVER_mean,TURNOVER_std))
            
            ret["past_days"].insert(0,feather_ret) 
            
        # 额外\t分割的current_date,stock_no,dataset_type, 便于后续数据集拆分、pair构造等
        datestock_uid = str(ret["current_date"]) + ret['stock_no']
        print("%s;%s;%s;0;%s" % (datestock_uid,ret["current_date"],ret['stock_no'],ret))

    def process_train_data(self):
        sql = "select * from stock_raw_daily where stock_no='%s' and TOPEN>0 order by trade_date desc"%(self.stock_no)
        df = pd.read_sql(sql, conn)
        
        end_idx = len(df) - self.past_days + 1
        for idx in range(self.future_days, end_idx):
            self.process_row(df, idx)
            # break
        
        df = None  # 释放内存？
        del df  
    
    def get_max_trade_date(self):
        trade_date = 0
        c = self.conn.cursor()
        cursor = c.execute("select max(trade_date) from stock_raw_daily;")
        for row in cursor:
            trade_date = row[0]
        cursor.close()
        return trade_date
    
    def process_predict_data(self):
        # max_trade_date = self.get_max_trade_date() #
        sql = "select * from stock_raw_daily where stock_no='%s' and TOPEN>0 order by trade_date desc limit 0,%d"%(self.stock_no,self.past_days)
        df = pd.read_sql(sql, conn)
        # print(df.loc[0]['trade_date'],max_trade_date)
        
        self.process_row(df, 0)
        
        df = None  # 释放内存？
        del df  
    
    def process(self):
        if self.data_type == "train":
            self.process_train_data()
        else:
            self.process_predict_data()

  
def process_all_stocks(data_type="train", processes_idx=-1):
    stocks = load_stocks()
    time_start = time.time()
    for i, stock in enumerate(stocks):
        p = PreProcessor(conn,stock[0],FUTURE_DAYS,PAST_DAYS,data_type)
        if processes_idx < 0 or data_type!="train": #predict耗时少，不用拆分
            p.process() 
        elif i % PROCESSES_NUM == processes_idx:
            p.process()
        
        # if i>1:
        #     break
        
    time_end = time.time() 
    time_c= time_end - time_start   #运行所花时间
    # print('time-cost:', time_c, 's') #predict=110s, train=157*400/60=17.5 hours ?
    
    # 解决生成速度慢的方案
    # 1. 并行
    # 2. 增量添加
    
    conn.close()

# python preprocess4rnn.py train 0 > data/rnn_train.txt_0 &
# python preprocess4rnn.py predict > data/rnn_predict.txt &
if __name__ == "__main__":
    data_type = sys.argv[1]
    process_idx = -1 if len(sys.argv) != 3 else int(sys.argv[2])
    # process_all_stocks(data_type, process_idx)
    
    p = PreProcessor(conn,"000001",3,20, data_type)
    p.process()
     
    conn.close() 
