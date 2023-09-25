import os
import sys
import json
import random
import pandas as pd
import sqlite3
from common import load_trade_dates

SAMPLE_COUNT_PER_DAY = 24  #每个交易日抽取多少条作为样本

conn = sqlite3.connect("file:data/stocks.db", uri=True)


def load_ids_by_date(date,dateset_type=0):
    sql = "select pk_date_stock from stock_for_transfomer where trade_date=%s and dataset_type=%d" %(date,dateset_type)
    df = pd.read_sql(sql, conn)
    return df['pk_date_stock']

def load_ids_by_date_label(date,dateset_type=0,label=1):
    sql = "select pk_date_stock from stock_for_transfomer where trade_date=%s and dataset_type=%d and list_label=%s " %(date,dateset_type,label)
    df = pd.read_sql(sql, conn)
    return df['pk_date_stock']

def upate_dataset_type(selected_ids,dataset_type):
    '''
    dataset_type: 1=验证集 2=测试集
    '''
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
    # label_cnt_map = [4,3,3,3,2,2,2,1]  
    cnt = 3 #3*8=24，每个档位抽3个，共计抽24个, 相当提升了高档位的占比，让模型更多关注高档位部分
    selected_ids = []
    for label in range(8): 
        df = load_ids_by_date_label(date,0,label+1)
        # cnt = label_cnt_map[label]
        if len(df) > cnt*2:
            c_ids = df.sample(n=cnt).values.tolist()
            selected_ids = selected_ids + c_ids
        else:
            print("error validate data cnt less date=%s label=%s"%(date,label))
    
    if len(selected_ids) > 0 :
        print(selected_ids)
        upate_dataset_type(selected_ids,1) #1=验证集
        
        
def test_dataset_split(date): 
    '''分割出测试集数据，抽样分布和整体分布近似'''
    df = load_ids_by_date(date,0) #取dataset_type=0(默认值)的数据
    if len(df) > SAMPLE_COUNT_PER_DAY*5:
        selected_ids = df.sample(n=SAMPLE_COUNT_PER_DAY).values.tolist()
        print(selected_ids)    
        upate_dataset_type(selected_ids,2) #2=测试集
    else:
        print("error test data cnt less",date)
             

def process_all():
    trade_dates = load_trade_dates(conn,1) #
    for date in trade_dates:  
        print(date)
        test_dataset_split(date)
        validate_dataset_split(date)
        # break 


# https://www.zditect.com/main-advanced/database/5-ways-to-run-sql-script-from-file-sqlite.html
# sqlite3 data/stocks.db ".read validate.sql"

if __name__ == "__main__":
    data_type = sys.argv[1] 
    if data_type == "all":
        process_all()
        # # python seq_data_split.py all  > all_seq_data_split.txt &
        # test_dataset_split(trade_dates)  
        # # run update sql 后，才能执行下一个? test/validate的先后执行顺序？
        # validate_dataset_split(trade_dates)
    if data_type == "validate":
        # python seq_data_split.py validate > validate.sql
        validate_dataset_split(20230815)
    if data_type == "test":
        # python seq_data_split.py test  #> test.sql
        test_dataset_split(20230815)
        
        