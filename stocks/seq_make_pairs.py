import os
import sys
import json
import random
import pandas as pd
import sqlite3
from common import load_stocks,load_trade_dates

PROCESSES_NUM = 5

conn = sqlite3.connect("file:data/stocks_train_2.db", uri=True)
            


# def load_ids_by_date(date,dateset_type=0):
#     sql = "select pk_date_stock from stock_for_transfomer where trade_date='%s' and dataset_type='%d'" %(date,dateset_type)
#     df = pd.read_sql(sql, conn)
#     return df['pk_date_stock']


def load_by_date(date,dateset_type=0,field="f_high_mean_rate"):
    sql = "select pk_date_stock,list_label,data_json from stock_for_transfomer where trade_date='%s' and dataset_type='%d'" %(date,dateset_type)
    df = pd.read_sql(sql, conn)
    
    # 因要两两构造pair对，加载jsons操作比较耗时，在这里做一次处理，可节省不少时间
    ret = []
    for index, row in df.iterrows():
        date_stock = row["pk_date_stock"]
        list_label = row["list_label"]
        obj = json.loads(row["data_json"])
        rate = obj[field]
        ret.append([date_stock,rate,list_label])
    return ret 

      
def load_by_stockno(stockno,dateset_type=0,field="f_high_mean_rate"):
    sql = "select pk_date_stock,list_label,data_json from stock_for_transfomer where stock_no='%s' and dataset_type='%d'" %(stockno,dateset_type)
    df = pd.read_sql(sql, conn)
    
    ret = []
    for index, row in df.iterrows():
        date_stock = row["pk_date_stock"]
        list_label = row["list_label"]
        obj = json.loads(row["data_json"])
        rate = obj[field]
        ret.append([date_stock,rate,list_label])
    return ret 


def make_pairs(rows):
    count = len(rows)
    for i in range(0,count-1):
        id_i = rows[i][0] # pk_date_stock
        # data_i = json.loads(rows.loc[i]["data_json"])
        rate_i = rows[i][1] 
        list_label_i = rows[i][2] 
        
        for j in range(i+1,count): 
            id_j = rows[j][0] #rows.loc[j]["pk_date_stock"]
            
            if id_i == id_j:
                continue 
            
            # rows[i].pk_date_stock
            # 两个比较，数值上有一定的差别才能构成算成pair对？
            # pk_date_stock_1,pk_date_stock_2,dataset_type(train|vaildate|test),field_1,field_2,field_3(0相等，1小于，2大于)
            
            # data_j = json.loads(rows.loc[j]["data_json"]) 
            rate_j = rows[j][1]  #data_j[field] 
            list_label_j = rows[j][2]
            
            # print(id_i,id_j) # 未过滤，190；
            # 0.01 ,过滤后129, 0.05 
            if abs(rate_i-rate_j)>0.15 and 8 in [list_label_j,list_label_i]: 
                # 在这里把pair的choose,reject排好序。
                if rate_i>rate_j:
                    # map(lambda x:str(x), [id_i,id_j,list_label_i,list_label_j]) 
                    print(";".join([str(x) for x in [id_i,id_j,list_label_i,list_label_j]])+";")
                else:
                    print(";".join([str(x) for x in [id_j,id_i,list_label_j,list_label_i]])+";")
                    
        #     if j>10:
        #         break #debug
        # break #debug
            

def process_pairs(dataset_type,pair_type="date",field="f_high_mean_rate",last_trade_date=20080808,process_idx=-1):
    if pair_type == "date":
        # 根据同一交易日下，不同股票构造pair对
        trade_dates = load_trade_dates(conn,0) # start_date=0
        for idx,date in enumerate(trade_dates):     
            # if date > 20220415:
            #     continue 
            
            if (process_idx < 0 or dataset_type!=0) or idx % PROCESSES_NUM == process_idx: #predict耗时少，不用拆分
                # print("a:",idx)
                data_rows = load_by_date(date,dataset_type,field)
                make_pairs(data_rows)
            
            # break #debug
            #增量的情况？新增一个交易日
    elif pair_type == "stock": 
        # 根据同一股票下，不同日期构造pair对
        conn1 = sqlite3.connect("file:data/stocks.db", uri=True)
        stocks = load_stocks(conn1)
        for idx,stock in enumerate(stocks):
            if (process_idx < 0 or dataset_type!=0) or idx % PROCESSES_NUM == process_idx: 
                # print("b:",idx)
                data_rows = load_by_stockno(stock[0],dataset_type,field)
                make_pairs(data_rows)
                
                # break #debug
                # 增量处理？
    else:
        pass 


# python seq_make_pairs.py 0 f_high_mean_rate 0 > f_high_mean_rate/train.txt_0 &
# python seq_make_pairs.py 0 f_high_mean_rate 1 > f_high_mean_rate/train.txt_1 &
# python seq_make_pairs.py 1 f_high_mean_rate> f_high_mean_rate/validate.txt &
# python seq_make_pairs.py 2 f_high_mean_rate> f_high_mean_rate/test.txt &
if __name__ == "__main__":                       
    dataset_type = int(sys.argv[1])
    pair_type = sys.argv[2]
    process_idx = -1 if len(sys.argv) != 4 else int(sys.argv[3])
    process_pairs(dataset_type,pair_type,process_idx=process_idx) #train=0,validate=1,test=2
    conn.close()
    
    # ret = load_by_date(20230801)
    # print(ret)
    
    # x = load_trade_dates(conn)
    # print(x)
    
    # date = 20230825
    # dataset_type = 0
    # field = "f_high_mean_rate"
    # data_rows = load_by_date(date,dataset_type,field)
    # make_pairs(data_rows)