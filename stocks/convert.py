import os
import sys
import json
import random
import pandas as pd
import sqlite3

conn = sqlite3.connect("file:data/stocks.db", uri=True)

# cat data/rnn_all_data.csv | python convert.py 

def save_to_kv():
    pass

def load_trade_dates():
    sql = "select distinct trade_date from stock_for_transfomer_test"
    df = pd.read_sql(sql, conn)
    return df['trade_date'].tolist()

def load_by_date(date,dateset_type=0):
    sql = "select pk_date_stock from stock_for_transfomer_test where trade_date='%s' and dataset_type='%d'" %(date,dateset_type)
    df = pd.read_sql(sql, conn)
    return df['pk_date_stock']

def convert():
    for i,line in enumerate(sys.stdin):
        # print(i,line)
        line = line.strip().replace("'",'"')
        obj = json.loads(line)
        # current_date,stock_no,dataset_type,data_json
        date = obj.get("current_date")
        stock_no = obj.get('stock_no')
        pk = str(date) + stock_no
        print("%s;%s;%s;0;%s" % (pk,date,stock_no,line))
        if i>500000:
            break

def make_pairs(rows):
    count = len(rows)
    for i in range(0,count-1):
        for j in range(i+1,count):
            # rows[i].pk_date_stock
            # 两个比较，数值上有一定的差别才能构成算成pair对？
            # pk_date_stock_1,pk_date_stock_2,dataset_type(train|vaildate|test),field_1,field_2,field_3(0相等，1小于，2大于)
            print(rows[i],rows[j])
            

def process_pairs(dataset_type):
    trade_dates = load_trade_dates()
    for date in trade_dates:
        stocks = load_by_date(date,dataset_type)
        make_pairs(stocks) 

def upate_dataset_type(selected_ids,dataset_type):
    str_selected_ids = ",".join([str(x) for x in selected_ids])
    sql = "update stock_for_transfomer_test set dataset_type=%d where pk_date_stock in (%s)" %(dataset_type,str_selected_ids)
    print(sql) 
    c = conn.cursor()
    cursor = c.execute(sql)
    conn.commit()
    cursor.close()

def dataset_split(dataset_type): # 1=验证集 2=测试集
    trade_dates = load_trade_dates()
    for date in trade_dates: 
        df = load_by_date(date,0)
        selected_ids = df.sample(n=20)# 
        upate_dataset_type(selected_ids,dataset_type)
         
        
# convert()
# make_pairs([0,1,2,3,4])
# load_trade_dates()
# df = load_by_date('20200318',0)
# print(df)

# dataset_split(1)
# dataset_split(2)

process_pairs(0) #train=0,validate=1,test=2

conn.close()
#  .import data/rnn_all_data_new.csv stock_for_transfomer