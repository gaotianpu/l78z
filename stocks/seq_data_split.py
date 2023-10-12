import os
import sys
import json
import random
import pandas as pd
import sqlite3
from common import load_trade_dates

SAMPLE_COUNT_PER_DAY = 24  #每个交易日抽取多少条作为样本

conn = sqlite3.connect("file:data/stocks_train.db", uri=True)

def load_ids_by_date(date,dateset_type=0,conn=conn):
    sql = "select pk_date_stock,list_label from stock_for_transfomer where trade_date=%s and dataset_type=%d" %(date,dateset_type)
    df = pd.read_sql(sql, conn)
    return df

def load_ids_by_date_label(date,dateset_type=0,label=1,conn=conn):
    sql = "select pk_date_stock from stock_for_transfomer where trade_date=%s and dataset_type=%d and list_label=%s " %(date,dateset_type,label)
    df = pd.read_sql(sql, conn)
    return df['pk_date_stock']

def upate_dataset_type(selected_ids,dataset_type,conn=conn):
    '''
    dataset_type: 1=验证集 2=测试集
    '''
    print("dt_%s;%s"%(dataset_type,selected_ids))
    # return 
    commit_id_list = [(dataset_type, sid) for sid in selected_ids] 
    cursor = conn.cursor()
    try:
        sql = "update stock_for_transfomer set dataset_type=? where pk_date_stock=?"
        cursor.executemany(sql, commit_id_list)  # commit_id_list上面已经说明
        conn.commit()
    except:
        print("exception")
        conn.rollback()

def validate_dataset_split(date):
    '''分割出验证集数据，抽样分布和整体不同，更关注头部数据'''
    label_cnt_map = [5,5,3,3,2,2,2,2] #注意顺序
    # cnt = 3 #3*8=24，每个档位抽3个，共计抽24个, 相当提升了高档位的占比，让模型更多关注高档位部分
    df_date = load_ids_by_date(date,0)

    selected_ids = []
    for label in range(8):  
        df = df_date[df_date['list_label']==(label+1)]
        cnt = label_cnt_map[label]
        if len(df) > cnt*2:
            c_ids = df["pk_date_stock"].sample(n=cnt).values.tolist()
            selected_ids = selected_ids + c_ids
        else:
            print("error validate data cnt less date=%s label=%s"%(date,label))
    
    if len(selected_ids) > 0 :
        upate_dataset_type(selected_ids,1) #1=验证集
        
        
def test_dataset_split(date): 
    '''分割出测试集数据，抽样分布和整体分布近似'''
    df = load_ids_by_date(date,0) #取dataset_type=0(默认值)的数据
    if len(df) > SAMPLE_COUNT_PER_DAY*5:
        selected_ids = df["pk_date_stock"].sample(n=SAMPLE_COUNT_PER_DAY).values.tolist()
        upate_dataset_type(selected_ids,2) #2=测试集
    else:
        print("error test data cnt less",date)
             

def process_all():
    trade_dates = load_trade_dates(conn,start_date=1) #
    for date in trade_dates:  
        print("date:",date)
        test_dataset_split(date)
        validate_dataset_split(date)
        # break 

def update_dataset_type_from_file():
    # sqlite3 导出 dataset_type 是1,2的脚本
    conn = sqlite3.connect("file:data/stocks_train.db", uri=True)
    with open("dataset_type_1.txt",'r') as f :
       upate_dataset_type([line.strip() for line in f.readlines()],1,conn=conn)
       
    with open("dataset_type_2.txt",'r') as f :
       upate_dataset_type([line.strip() for line in f.readlines()],2,conn=conn)

# https://www.zditect.com/main-advanced/database/5-ways-to-run-sql-script-from-file-sqlite.html
# sqlite3 data/stocks.db ".read validate.sql"

def c_round(x):
    return round(x,4)

def gen_next_day_data(dataset_type=2):
    sql = 'select pk_date_stock,trade_date,stock_no from stock_for_transfomer where dataset_type=%s order by trade_date'%(dataset_type)
    df = pd.read_sql(sql, conn)
    
    df_all = None
    for idx,row in df.iterrows():
        # print(row['pk_date_stock'])
        sql = "select * from stock_raw_daily_2 where stock_no='%s' and trade_date>%s order by trade_date asc limit 1" % (row['stock_no'],row['trade_date'])
        df_p = pd.read_sql(sql, conn)
        df_p["pk_date_stock"] = row['pk_date_stock']
        # print(idx,row['stock_no'],row['trade_date'])
        # print(df_p)
        if idx==0:
            df_all = df_p
        else:
            df_all = pd.concat([df_all,df_p])
        
        # if idx>2:
            # break
    
    # print(df_all)
    df_all['change_rate'] = df_all.apply(lambda x: x['change_rate']/100, axis=1) 
    df_all['last_close'] = df_all['CLOSE_price'] - df_all['change_amount'] 
    df_all['open_rate'] = c_round((df_all['OPEN_price'] - df_all['last_close']) / df_all['last_close']) 
    df_all['low_rate'] = c_round((df_all['LOW_price'] - df_all['last_close']) / df_all['last_close']) 
    df_all['high_rate'] = c_round((df_all['HIGH_price'] - df_all['last_close']) / df_all['last_close']) 
    
    df_all.rename(columns={'trade_date':'trade_date_next'},inplace=True) #列重命名
    df_all.to_csv("data/test_stock_raw_daily_3.txt",sep=";",index=False)  



def export_by_label(): #list_label,dataset_type=0
    conn1 = sqlite3.connect("file:data/stocks_train_2.db", uri=True)
    for dataset_type in range(3):
        for label in range(8):
            list_label = label + 1
            print(dataset_type,list_label)
            sql = "select pk_date_stock from stock_for_transfomer where dataset_type=%s and list_label=%s" % (dataset_type,list_label)
            df = pd.read_sql(sql, conn1)
            df.to_csv("data2/point_%s_%s.txt"%(dataset_type,list_label),sep=";", header=None,index=False)
            # break 
        # break

if __name__ == "__main__":
    data_type = sys.argv[1] 
    print(data_type)
    if data_type == "all":
        process_all()
        # python seq_data_split.py all  > all_seq_data_split_v2
        # # python seq_data_split.py all  > all_seq_data_split.txt &
        # test_dataset_split(trade_dates)  
        # # run update sql 后，才能执行下一个? test/validate的先后执行顺序？
        # validate_dataset_split(trade_dates)
    if data_type == "update_from_file":
        # python seq_data_split.py update_from_file
        update_dataset_type_from_file()
    if data_type == "validate":
        # python seq_data_split.py validate > validate.sql
        validate_dataset_split(20230815)
    if data_type == "test":
        # python seq_data_split.py test  #> test.sql
        test_dataset_split(20230815)
    if data_type == "next_day":
        # python seq_data_split.py next_day  
        gen_next_day_data(2)
    
    if data_type == "convert_stock_raw_daily":
        # start_date=20230928
        # python seq_data_split.py convert_stock_raw_daily
        convert_stock_raw_daily()
        # cat data/trade_dates/*.txt > data/stock_raw_daily_4.txt
    
    if data_type == "export_by_label":
        export_by_label()
        
        