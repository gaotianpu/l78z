import os
import sys
import json
import random
import pandas as pd
import sqlite3

conn = sqlite3.connect("file:data/stocks.db", uri=True)

def load_stocks():
    with open("schema/stocks.txt", 'r') as f:
        l = []
        for i, line in enumerate(f):
            fields = line.strip().split(',')
            # 0:stock_no,1:start_date,2:stock_name,3:是否退市
            if fields[3] == "1":
                continue
            yield fields
            
def load_trade_dates():
    sql = "select distinct trade_date from stock_for_transfomer"
    df = pd.read_sql(sql, conn)
    return df['trade_date'].tolist()

def load_ids_by_date(date,dateset_type=0):
    sql = "select pk_date_stock from stock_for_transfomer where trade_date='%s' and dataset_type='%d'" %(date,dateset_type)
    df = pd.read_sql(sql, conn)
    return df['pk_date_stock']

def load_idjson_by_date(date,dateset_type=0):
    sql = "select pk_date_stock,data_json from stock_for_transfomer where trade_date='%s' and dataset_type='%d'" %(date,dateset_type)
    df = pd.read_sql(sql, conn)
    return df

def load_idjson_by_stockno(stockno,dateset_type=0):
    sql = "select pk_date_stock,data_json from stock_for_transfomer where stock_no='%s' and dataset_type='%d'" %(stockno,dateset_type)
    df = pd.read_sql(sql, conn)
    return df

# def convert():
#     for i,line in enumerate(sys.stdin):
#         line = line.strip().replace("'",'"')
#         obj = json.loads(line)
#         # current_date,stock_no,dataset_type,data_json
#         date = obj.get("current_date")
#         stock_no = obj.get('stock_no')
#         pk = str(date) + stock_no
#         print("%s;%s;%s;0;%s" % (pk,date,stock_no,line))
#         # if i>500000: #小数据测试,debug
#         #     break

def make_pairs(rows,field="f_mean_rate"):
    count = len(rows)
    for i in range(0,count-1):
        for j in range(i+1,count):
            id_i = rows.loc[i]["pk_date_stock"]
            id_j = rows.loc[j]["pk_date_stock"]
            
            if id_i == id_j:
                continue 
            
            # rows[i].pk_date_stock
            # 两个比较，数值上有一定的差别才能构成算成pair对？
            # pk_date_stock_1,pk_date_stock_2,dataset_type(train|vaildate|test),field_1,field_2,field_3(0相等，1小于，2大于)
            data_i = json.loads(rows.loc[i]["data_json"].replace("'",'"'))
            data_j = json.loads(rows.loc[j]["data_json"].replace("'",'"'))
            rate_i = data_i[field]
            rate_j = data_j[field] 
            
            # print(id_i,id_j) # 未过滤，190；
            if abs(rate_i-rate_j)>0.01: #过滤后129
                # 在这里把pair的choose,reject排好序。
                if rate_i>rate_j:
                    print(id_i,id_j) 
                else:
                    print(id_j,id_i)
                    
        #     if j>10:
        #         break #debug
        # break #debug
            

def process_pairs(dataset_type,field="f_mean_rate",last_trade_date=20080808):
    # 根据同一交易日下，不同股票构造pair对
    trade_dates = load_trade_dates()
    for date in trade_dates:
        # if date<last_trade_date:
        #     continue 
        data_rows = load_idjson_by_date(date,dataset_type)
        make_pairs(data_rows,field)
    #     break #debug
    #增量的情况？新增一个交易日
        
    # 根据同一股票下，不同日期构造pair对
    stocks = load_stocks()
    for stock in stocks:
        data_rows = load_idjson_by_stockno(stock[0],dataset_type)
        make_pairs(data_rows,field)
        # break #debug
        # 增量处理？


# python seq_make_pairs.py 0 f_mean_rate> f_mean_rate/train.txt &
# python seq_make_pairs.py 1 f_mean_rate> f_mean_rate/validate.txt &
# python seq_make_pairs.py 2 f_mean_rate> f_mean_rate/test.txt &
if __name__ == "__main__":
    data_type = int(sys.argv[1])
    process_pairs(data_type) #train=0,validate=1,test=2
    conn.close()