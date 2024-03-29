#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import time
import sqlite3  
import pandas as pd
import json
from common import load_stocks,c_round,load_trade_dates,get_max_trade_date

MAX_ROWS_COUNT = 2000 #从数据库中加载多少数据, 差不多8年的交易日数。
conn = sqlite3.connect("file:newdb/stocks.db?mode=ro", uri=True)

forecast_fields = 'f_high_mean_rate,f_high_rate,f_low_mean_rate,f_low_rate'.split(",")
static_fields = 'mean,std,25%,50%,75%'.split(",")
daily_fields = "OPEN_price,CLOSE_price,change_amount,change_rate,LOW_price,HIGH_price,TURNOVER,TURNOVER_amount,TURNOVER_rate,low_rate,high_rate".split(",")
daily_fields = "open_rate,low_rate,high_rate,change_rate".split(",")

def get_all_fields():
    fields = ['key','count']
    for f1 in forecast_fields:
        for f2 in static_fields:
            fields.append("%s_%s"%(f1,f2))
    return fields

#  select * from stock_raw_daily order by RANDOM() limit 2;
# select stock_no,round(avg(TURNOVER_rate),2) from stock_raw_daily where stock_no in (300879,300880,300881,300882,300883,300884,300885,300886,300887,300889,300892,300893,300894,300895,300896,300897,300898,300899,300900,300901,300902,300910,300911,300912,300913,300915,300916,300919,300925,300928,300929,300935,300939,300940,300946,300948,300949,300951,300952,300953,300958,300959,300960,300961,300968,300971,300973,300999,301000,301001) and  TURNOVER_rate<>'-' group by stock_no;
# update stock_raw_daily set TURNOVER_rate=%s where stock_no='%s' and TURNOVER_rate='-';


def get_update_sql():
    sql = "select stock_no,round(avg(TURNOVER_rate),2) as TURNOVER_rate from stock_raw_daily where stock_no in ('000596','000796','001914','002074','002789','002798','002801','002809','002819','002820','002821','002840','002893','002899','002900','002901','002923','002926','002937','002939','002950','002952','002953','002959','002961','002962','002963','002965','002966','002973','002976','002984','002987','002988','002990','002991','002992','002993','002997','002999','003000','003003','003006','003011','003012','003015','003016','003017','003018','003019','003026','003027','003030','003035','003036','003037','003040','003816') and  TURNOVER_rate<>'-' group by stock_no;"
    df = pd.read_sql(sql, conn)
    # df.loc[idx]['trade_date']
    for index, row in df.iterrows():
        usql = ("update stock_raw_daily set TURNOVER_rate=%s where stock_no='%s' and TURNOVER_rate='-';" %(row["TURNOVER_rate"],row["stock_no"]))
        print(usql)

def fix_TURNOVER_rate():
    conn = sqlite3.connect("file:newdb/stocks.db", uri=True)
    commit_id_list = []
    with open('uncollect_stock_no.txt','r') as f:
        for line in f:
            fields = line.strip().split(',') 
            stock_no = fields[0]
            # 
            sql = "select stock_no,round(avg(TURNOVER_rate),2) as TURNOVER_rate from stock_raw_daily where stock_no='%s' and  TURNOVER_rate<>'-'"%(stock_no)
            # sql = "select TURNOVER_rate from stock_raw_daily where stock_no='%s' and  TURNOVER_rate<>'-'"%(stock_no)
            # sql = "select distinct stock_no from stock_raw_daily where TURNOVER_rate='-'"
            # sql = "select trade_date,stock_no from stock_raw_daily where stock_no='%s' and  TURNOVER_rate='-'"%(stock_no)
            df = pd.read_sql(sql, conn)
            # if len(df)>0:
            #     for index, row in df.iterrows():
            #         print(row['trade_date'],row['stock_no'])
            commit_id_list.append( ( df.loc[0]['TURNOVER_rate'],df.loc[0]['stock_no'] ) )
            # break
    print(commit_id_list)    
    # commit_id_list = [(dataset_type, sid) for sid in selected_ids] 
    cursor = conn.cursor()
    try:
        sql = "update stock_raw_daily set TURNOVER_rate=? where stock_no=? and TURNOVER_rate='-';"
        cursor.executemany(sql, commit_id_list)  # commit_id_list上面已经说明
        conn.commit()
    except:
        print("exception")
        conn.rollback()

def map_val_range(val,val_ranges):
    for i,val_range in enumerate(val_ranges):
        if val<val_range:
            return i+1
    return len(val_ranges) + 1

# pk_date_stock UNSIGNED BIG INT NOT NULL,
#     trade_date  INT    NOT NULL,
#     stock_no    CHAR(6)    NOT NULL,
#     dataset_type TINYINT NOT NULL DEFAULT 0,   
#     data_json   TEXT    NOT NULL,
    
def map_val_range_all():
    val_ranges = [-0.00493,0.01434,0.02506,0.03954,0.04997,0.06524,0.09353]
    stocks = load_stocks(conn)
    for stock in stocks:
        stock_no = stock[0] 
        sql = "select * from stock_for_transfomer where stock_no='%s' limit 10;" %(stock_no)
        df = pd.read_sql(sql, conn)
        for idx,row in df.iterrows():
            data_json = json.loads(row['data_json'])
            val = data_json.get("f_high_mean_rate")
            val_label = map_val_range(val,val_ranges)
            print(row['pk_date_stock'],row['trade_date'],row['stock_no'],row['dataset_type'],val_label,val)
        break 

def sample_statics(stocks): 
    # 结果： [-0.00493,0.01434,0.02506,0.03954,0.04997,0.06524,0.09353]
    #total=6606760 #350000  order by RANDOM()
    # select * from stock_for_transfomer  order by RANDOM() limit 350000;
    li_ = []
    for stock in stocks:
        stock_no = stock[0] 
        sql = "select * from stock_for_transfomer where stock_no='%s' order by RANDOM() limit 128;" %(stock_no)
        df = pd.read_sql(sql, conn)
        for idx,row in df.iterrows():
            data_json = json.loads(row['data_json'])
            val = data_json.get("f_high_mean_rate")
            li_.append(val) 
    
    li_.sort()
    cnt = len(li_)
    range_pct = [0.25,0.5,0.625,0.75,0.8125,0.875,0.9375]
    range_idx = [int(cnt*pct)-1 for pct in range_pct]
    range_val = [li_[idx] for idx in range_idx]
    # print(li_)
    # print(range_idx)
    print(range_val)   


class StockStatics():
    def __init__(self,stocks,conn):
        self.static_fields = 'mean,std,25%,50%,75%,min,max'.split(",")
        self.daily_fields = "open_rate,low_rate,high_rate,change_rate".split(",")
        self.all_fields = self.get_static_fields() 
        self.stocks = stocks
        self.conn = conn
    
    #生成字段名列表
    def get_static_fields(self):
        all_fields = ['stock_no','open_rate_label','count'] #max_trade_date
        for field in daily_fields:
            for f2 in static_fields: 
                all_fields.append("%s_%s" %(field,f2)) 
        return all_fields

    def compute_mean_std(self,stock_no):
        '''抽样计算整体的均值和标准差'''
        sql = "select * from stock_raw_daily where stock_no='%s' "%(stock_no)
        df = pd.read_sql(sql, self.conn)

        # if stock_no=="002913":
        #     df['change_rate'] = df['change_rate'].str.replace("%","")
        #     df['change_rate'] = pd.to_numeric(df['change_rate']) #.astype('double')
        #     df['TURNOVER_rate'] = df['TURNOVER_rate'].str.replace("%","")
        #     df['TURNOVER_rate'] = pd.to_numeric(df['TURNOVER_rate'])
        
        df['last_close_price'] = df['CLOSE_price'] - df['change_amount'] 
        df['open_rate'] = c_round((df['OPEN_price'] - df['last_close_price']) / df['last_close_price']) 
        df['low_rate'] = c_round((df['LOW_price'] - df['last_close_price']) / df['last_close_price']) 
        df['high_rate'] = c_round((df['HIGH_price'] - df['last_close_price']) / df['last_close_price']) 
        df['change_rate'] = df.apply(lambda x: x['change_rate']/100, axis=1)
            
        all_ret = []
        
        # 整体统计
        df = df[daily_fields]
        df_describe = df.describe()   
        ret = [stock_no,0,len(df)]
        for field in daily_fields:
            for f2 in static_fields:
                ret.append(c_round(df_describe[field][f2]))
        all_ret.append(ret)
        
        # 根据open_rate 确定label数值
        df_statics_stock = dict(zip(self.all_fields,ret))
        (or25,or50,or75) = (df_statics_stock['open_rate_25%'],df_statics_stock['open_rate_50%'],df_statics_stock['open_rate_75%'])
        df['open_rate_label'] = df.apply(lambda x: 1 if x['open_rate'] < or25 else 2 if x['open_rate'] < or50 else 3 if x['open_rate'] < or75 else 4, axis=1)
        # df.to_csv("data/static_seq_stocks_tmp.txt",sep=";",index=False)   
        
        # 基于 open_rate_label 统计
        label_groups = df.groupby('open_rate_label')
        for open_rate_label,data in label_groups: 
            ret = [stock_no,open_rate_label,len(data)]
            df = data[daily_fields]
            df_describe = df.describe()  
            # print(df_describe)
            for field in self.daily_fields:
                for f2 in self.static_fields:
                    ret.append(c_round(df_describe[field][f2]))
            all_ret.append(ret)
        
        # for ret in all_ret:
        #     print(ret)

        return all_ret
            

    def process(self):
        '''计算每个stocks，价格，成交量，成交金额等字段的均值、标准差''' 
        li = []
        for i, stock in enumerate(self.stocks):
            stock_no = stock[0]
            # print(stock_no)
            li = li + self.compute_mean_std(stock_no)
            # try:
            #     li.append(compute_mean_std(stock_no))
            # except:
            #     print("error:",stock_no)
            # break 
                
        df = pd.DataFrame(li,columns=self.all_fields)
        df.to_csv("data/static_seq_stocks.txt",sep=";",index=False)  

def test():
    fields = "OPEN_price,CLOSE_price,change_amount,change_rate,LOW_price,HIGH_price,TURNOVER,TURNOVER_amount,TURNOVER_rate,last_close,open_rate,low_rate,high_rate,high_low_range,open_low_rate,open_high_rate,open_close_rate,TURNOVER_idx,TURNOVER_amount_idx,TURNOVER_rate_idx,change_rate_idx,last_close_idx,open_rate_idx,low_rate_idx,high_rate_idx,high_low_range_idx,open_low_rate_idx,open_high_rate_idx,open_close_rate_idx".split(",")
    static_fields = 'mean,std,25%,50%,75%,max,min'.split(",")
    stocks = load_stocks(conn) 
    for i, stock in enumerate(stocks):
        stock_no = stock[0]
        sql = "select * from stock_raw_daily where stock_no='%s'"%(stock_no)
        df = pd.read_sql(sql, conn)
        describe = df.describe() 
        trade_date_max = df['trade_date'].max()
        
        d = {'stock_no':stock_no,'trade_date_max':trade_date_max}
        # static_li = [stock_no]
        for f1 in fields:
            for f2 in static_fields:
                d['%s_%s'%(f1,f2)] = round(describe[f1][f2],4) 
                # static_li.append(round(describe[f1][f2],4))
        # print(static_li)
        print("%s;%s;%s" %(stock_no,trade_date_max,json.dumps(d)))
        # break
        

def tmp():
    stocks = load_stocks(conn)
    time_start = time.time()
    for i, stock in enumerate(stocks):
        stock_no = stock[0]
        sql = "select * from stock_raw_daily where stock_no='%s' and TOPEN>0 order by trade_date desc limit 0,%d"%(stock_no,MAX_ROWS_COUNT)
        df = pd.read_sql(sql, conn)
        
        last_trade_date = df["trade_date"].max()
        
        # 统计价格、成交量、成交金额、换手率等的均值和标准差，用于后续的归一化处理
        df_history_describe = df.describe() 
        # 价格
        price_mean = c_round(df_history_describe["TOPEN"]["mean"])
        price_std = c_round(df_history_describe["TOPEN"]["std"]) 
        # VOTURNOVER 成交金额
        VOTURNOVER_mean = c_round(df_history_describe["VOTURNOVER"]["mean"])
        VOTURNOVER_std = c_round(df_history_describe["VOTURNOVER"]["std"])
        # VATURNOVER 成交量
        VATURNOVER_mean = c_round(df_history_describe["VATURNOVER"]["mean"])
        VATURNOVER_std = c_round(df_history_describe["VATURNOVER"]["std"])
        #  ('TURNOVER', '换手率')
        TURNOVER_mean = c_round(df_history_describe["TURNOVER"]["mean"])
        TURNOVER_std = c_round(df_history_describe["TURNOVER"]["std"]) 
        
        ret ={"price_mean":price_mean,"price_std":price_std,"VOTURNOVER_mean":VOTURNOVER_mean,"VOTURNOVER_std":VOTURNOVER_std,
            "VATURNOVER_mean":VATURNOVER_mean,"VATURNOVER_std":VATURNOVER_std,"TURNOVER_mean":TURNOVER_mean,"TURNOVER_std":TURNOVER_std}
        
        print("%s;%s;%s"%(stock_no,str(last_trade_date),json.dumps(ret))) 


def static_seq(bykey,sql):
    df = pd.read_sql(sql, conn)
    
    li_ = []
    for idx,row in df.iterrows():
        data_json = json.loads(row['data_json'])
        ret = [data_json.get(field) for field in forecast_fields]
        # print(ret) 
        li_.append(ret)
    
    df_v = pd.DataFrame(li_,columns=forecast_fields)
    df_v_d = df_v.describe() 
    # print(df_v_d)
    
    static_li = [bykey,int(df_v_d['f_high_mean_rate']["count"])]
    for f1 in forecast_fields:
        for f2 in static_fields:
            static_li.append(round(df_v_d[f1][f2],4))
            
    # print(";".join( [str(x) for x in static_li])) 
    
    return static_li

def static_forecast_val():
    # 按天/股票统计最大值，最小值的均值？
    t_li = []
    dates = load_trade_dates(conn,0)
    for trade_date in dates:
        sql = "select * from stock_for_transfomer where trade_date=%s" % (trade_date) 
        t_li.append(static_seq(trade_date,sql)) 
    df_v = pd.DataFrame(t_li,columns=get_all_fields())
    df_v.to_csv("data/static_seq_dates.txt",sep=";",index=False)  
    
    # stock_no='002913' 
    
    s_li = []
    stocks = load_stocks(conn)
    for stock in stocks:
        stock_no = stock[0]
        sql = "select * from stock_for_transfomer where stock_no='%s'" % (stock_no) 
        try:
            s_li.append(static_seq(stock_no,sql))  
        except:
            print("error:", stock_no)
    df_v = pd.DataFrame(s_li,columns=get_all_fields())
    df_v.to_csv("data/static_seq_stocks.txt",sep=";",index=False)

def compute_buy_price(df_predict=None):
    select_cols='stock_no,low_rate_25%,low_rate_50%,low_rate_75%,high_rate_25%,high_rate_50%,high_rate_75%'.split(",")
    
    if not df_predict:
        df_predict = pd.read_csv("data/predict/predict_merged.txt",sep=";",header=0,dtype={'stock_no': str})
    
    trade_date = str(df_predict['pk_date_stock'].values[0])[:8]
    
    df = df_predict.merge(
        pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0,dtype={'stock_no': str})[select_cols],
        on="stock_no",how='left')
    for col in select_cols:
        if col == "stock_no":
            continue
        fieldname = 'price_' + col
        df[fieldname] = df.apply(lambda x: x['CLOSE_price'] * (1 + x[col]), axis=1)
        df[fieldname] = df[fieldname].round(2) 
    
    df.to_csv("data/predict/predict_buy_price.txt.%s"%(trade_date),sep=";",index=False) 
    df.to_csv("predict_buy_price.txt",sep=";",index=False) 

def hold_days(stock_no):
    data_file = "data/hold_days/%s.txt"%(stock_no)
    
    sql = "select * from stock_raw_daily where stock_no='%s' and OPEN_price>0 order by trade_date asc"%(stock_no)
    df = pd.read_sql(sql, conn) 
    count = len(df)
    li = []
    for idx in range(count):
        start_date = int(df.loc[idx]['trade_date'])
        start_OPEN_price = df.loc[idx]['OPEN_price']
        for day in range(2,11): 
            if idx+day>count:
                break 
            max_price = df.loc[idx+1:idx+day]['HIGH_price'].max()
            min_price = df.loc[idx+1:idx+day]['LOW_price'].min()
            max_rate = c_round((max_price - start_OPEN_price)/start_OPEN_price)
            min_rate = c_round((min_price - start_OPEN_price)/start_OPEN_price)
            li.append([stock_no,start_date,day,max_rate,min_rate])
    df_s = pd.DataFrame(li,columns='stock_no,start_date,day,max_rate,min_rate'.split(',')) 
    df_s.to_csv(data_file,sep=";",index=False) 
    
    vali = [stock_no]
    for day in range(2,11):
        for field in ['max_rate','min_rate']:
            vali.append(c_round(df_s[df_s["day"]==day][field].mean()))
            vali.append(c_round(df_s[df_s["day"]==day][field].median()))
    print( ";".join([str(v) for v in vali]))  

def hold_days_all():
    stocks = load_stocks(conn)
    
    fields = ['stock_no']
    for day in range(2,11):
        for field in ['max_rate','min_rate']:
            fields.append("%s_%s_mean"%(day,field))
            fields.append("%s_%s_median"%(day,field))
    print( ";".join(fields))  
            
    for stock in stocks:
        stock_no = stock[0]
        hold_days(stock_no)
        # break 

def v2_f2():
    val_ranges = [-1.400000e-03,1.380000e-02,2.280000e-02,3.540000e-02,4.470000e-02,5.840000e-02,0.068900,0.084600,0.110700,0.122000,0.138700,0.169200]
    conn = sqlite3.connect("file:data3_f2/stocks_train_v2_f2.db?mode=ro", uri=True)
    #high_rate,low_rate,high1_rate,low1_rate
    sql = "select pk_date_stock,high_rate from stock_for_transfomer"
    df = pd.read_sql(sql, conn)
    df['list_label'] = df.apply(lambda x: len([val for val in val_ranges if x['high_rate']>val]) , axis=1)
    # for idx,row in df.iterrows():
    #     list_label = int(row["list_label"])
    #     pk_date_stock = int(row["pk_date_stock"])
    #     sql = f'update stock_for_transfomer set list_label={list_label} where pk_date_stock={pk_date_stock};'
    #     print(sql)
        # if idx>3:
        #     break 
    df.to_csv("data3_f2/high_rate_value.txt",sep=";",index=False) 
    # print(df.describe())
    # df = df[df['high_rate']>0.110700]
    # print(df.describe())
    
    

# python seq_statistics.py stocks #> stocks_statistics.jsonl
if __name__ == "__main__":
    op_type = sys.argv[1]
    # print(op_type)
    if op_type == "v2_f2":
        v2_f2()
        
    if op_type == "stocks":
        # python statistics.py stocks # data/static_seq_stocks.txt
        stocks = load_stocks(conn) #
        ss = StockStatics(stocks,conn)
        ss.process()
        # compute_per_stocks_mean_std()  
        # compute_mean_std('600008')
    if op_type == "dates":
        # python statistics.py dates # per_day_mean_std.jsonl
        static_forecast_val()
    if op_type == "buy_price":
        # python statistics.py buy_price # predict_buy_price.txt
        compute_buy_price()
    if op_type == "hold_days":
        # python statistics.py hold_days > hold_days_all.txt
        hold_days_all()
    if op_type == "sample_statics":
        # python statistics.py sample_statics > sample_statics.txt
        stocks = load_stocks(conn)
        for i in range(10):
            sample_statics(stocks)
    if op_type == "map_val_range":
        # python statistics.py map_val_range 
        map_val_range_all()
        # print(map_val_range(-0.0050))
        # print(map_val_range(0.0142))
        # print(map_val_range(0.0250))
        # print(map_val_range(0.0394))
        # print(map_val_range(0.0498))
        # print(map_val_range(0.0651))
        # print(map_val_range(0.0934))
        # print(map_val_range(0.0936))
        #val_ranges = [-0.0049, 0.0143, 0.0251, 0.0395, 0.0499, 0.0652, 0.0935]
    if op_type == "fix_TURNOVER_rate":
        fix_TURNOVER_rate()
    
    if op_type == "test":  
        # python statistics.py test > data/stock_statics.txt  
        test()
    
    # print(get_all_fields())
    
    # compute_mean_std("002913") # 601998,  002913
    
    # compute_mean_std()
    # get_update_sql() 