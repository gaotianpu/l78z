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

FUTURE_DAYS = 5 # 预测未来几天的数据
PAST_DAYS = 20 #使用过去几天的数据做特征

MAX_ROWS_COUNT = 1000 #从数据库中加载多少数据

max_high,min_high,max_low,min_low = 0.818,-0.6,0.668,-0.6

conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)

def zscore(x,mean,std):
    return round((x-mean)/(std+0.00000001),3)

class PreProcessor:
    def __init__(self, conn,stock_no, future_days = 3, past_days = 20):
        self.conn = conn
        self.stock_no = stock_no
        self.future_days = future_days
        self.past_days = past_days
    
    def process_row(self, df, idx):
        ret = {"stock_no": self.stock_no, "current_date":df.loc[idx]['trade_date']}
        
        # 未来值,FUTURE_DAYS最高价，最低价？
        if idx>0: #train
            base = df.loc[idx-1]['TOPEN'] # df_future.iloc[-1]['TOPEN']
            df_future = df.loc[idx-self.future_days : idx-2] #由于t+1,计算买入后第二个交易的价格
            highest = df_future['HIGH'].max()
            lowest = df_future['LOW'].min()
            mean = df_future['HIGH'].mean()
            
            # print(df_future.describe()) 
            # print(base,highest,lowest,mean)
            
            f_high_rate = round((highest-base)/base,2)
            f_low_rate = round((lowest-base)/base,2)
            f_mean_rate = round((mean-base)/base,2)
            
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
            
        print(ret)

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
    
    def process(self,type):
        if type == "train":
            self.process_train_data()
        else:
            self.process_predict_data()

  
def process_all_stocks(data_type="train"):
    stocks = load_stocks()
    
    time_start = time.time()
    for _, stock in enumerate(stocks):
        p = PreProcessor(conn,stock[0],3,20)
        p.process(data_type)
    
    time_end = time.time() 
    time_c= time_end - time_start   #运行所花时间
    print('time-cost:', time_c, 's') #predict=110s, train=
    
    conn.close()

# python preprocess4rnn.py train > data/rnn_train.txt &
# python preprocess4rnn.py predict > data/rnn_predict.txt &
if __name__ == "__main__":
    data_type = sys.argv[1]
    process_all_stocks(data_type)
    # p = PreProcessor(conn,"000001",3,20)
    # p.process(data_type) 
