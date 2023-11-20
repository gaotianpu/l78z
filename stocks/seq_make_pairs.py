import os
import sys
import json
import random
import pandas as pd
import sqlite3
from common import load_stocks,load_trade_dates

PROCESSES_NUM = 5

MIN_RATE = 0.06
DIFF_RATE = 0.06

conn = sqlite3.connect("file:data3/stocks_train_v3.db", uri=True)

def load_by_date(date,dateset_type=0,field="high_rate"):
    sql = f"select pk_date_stock,list_label,{field} from stock_for_transfomer where trade_date={date} and dataset_type={dateset_type}"
    df = pd.read_sql(sql, conn)
    
    # 因要两两构造pair对，加载jsons操作比较耗时，在这里做一次处理，可节省不少时间
    ret = []
    for index, row in df.iterrows():
        date_stock = int(row["pk_date_stock"])
        list_label = int(row["list_label"])
        # obj = json.loads(row["data_json"])
        rate = row[field]
        ret.append([date_stock,rate,list_label])
    return ret 

      
def load_by_stockno(stockno,dateset_type=0,field="high_rate"):
    sql = f"select pk_date_stock,list_label,{field} from stock_for_transfomer where stock_no='{stockno}' and dataset_type={dateset_type}"
    df = pd.read_sql(sql, conn)
    
    ret = []
    for index, row in df.iterrows():
        date_stock = row["pk_date_stock"]
        list_label = row["list_label"]
        # obj = json.loads(row["data_json"])
        rate = row[field]
        ret.append([date_stock,rate,list_label])
    return ret 


def make_pairs(rows):
    count = len(rows)
    for i in range(0,count-1):
        id_i = rows[i][0] # pk_date_stock
        rate_i = rows[i][1] 
        list_label_i = rows[i][2]
        
        for j in range(i+1,count): 
            id_j = rows[j][0] #rows.loc[j]["pk_date_stock"]
            
            rate_j = rows[j][1]
            list_label_j = rows[j][2]
            
            # DIFF_RATE: 两个比较，数值上有一定的差别才能构成算成pair对？
            # MIN_RATE: 如果两个数值均小于MIN_RATE，不再关注二者大小
            if (id_i == id_j) or (rate_i<MIN_RATE and rate_j<MIN_RATE) or (abs(rate_i-rate_j)<DIFF_RATE) :
                continue 
            
            # pk_date_stock_1,pk_date_stock_2,dataset_type(train|vaildate|test),field_1,field_2,field_3(0相等，1小于，2大于)
             
            
            # print(id_i,id_j) # 未过滤，190；
            # 0.01 ,过滤后129, 0.05 
            
            # 在这里把pair的choose,reject排好序。
            if rate_i>rate_j:
                # map(lambda x:str(x), [id_i,id_j,list_label_i,list_label_j]) 
                print(";".join([str(x) for x in [id_i,id_j,list_label_i,list_label_j]]))
            else:
                print(";".join([str(x) for x in [id_j,id_i,list_label_j,list_label_i]]))
                    
        #     if j>10:
        #         break #debug
        # break #debug
            

def process_pairs(dataset_type,pair_type="date",field="high_rate",process_idx=-1):
    if pair_type == "date":
        # 根据同一交易日下，不同股票构造pair对
        trade_dates = load_trade_dates(conn,0) # start_date=0
        for idx,date in enumerate(trade_dates):     
            # if date > 202309226: #暂时不用
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

# python seq_make_pairs.py 1 date > data3/pair.validate.date.txt &
# python seq_make_pairs.py 2 date > data3/pair.test.date.txt &
# python seq_make_pairs.py 0 date 0 > data3/pair.train.date.txt_0 &
if __name__ == "__main__":                       
    dataset_type = int(sys.argv[1])
    pair_type = sys.argv[2]
    process_idx = -1 if len(sys.argv) != 4 else int(sys.argv[3])
    process_pairs(dataset_type,pair_type,field="high_rate",process_idx=process_idx) #train=0,validate=1,test=2
    conn.close()