import os
import sys
import json
import random
import pandas as pd
import sqlite3
from common import load_trade_dates,c_round

conn = sqlite3.connect("file:newdb/stocks.db", uri=True)


def upate_db(commit_id_list,conn=conn):
    '''
    dataset_type: 1=验证集 2=测试集
    '''
    # print("dt_%s;%s"%(dataset_type,selected_ids))
    # return 
    # commit_id_list = selected_ids #[(dataset_type, sid) for sid in selected_ids] 
    cursor = conn.cursor()
    try:
        sql = "update stock_raw_daily set TURNOVER_amount=? where trade_date=? and stock_no=?"
        cursor.executemany(sql, commit_id_list)  # commit_id_list上面已经说明
        conn.commit()
    except:
        print("exception")
        conn.rollback()

def process():
    with open('newdb/stock_raw_daily.txt','r') as f:
        i = 0 
        rows = []
        for line in f: 
            fields = line.strip().split("|")  
            trade_date =  int(fields[0])
            stock_no =  fields[1] 
            TURNOVER_amount = float(fields[9])/10000
            rows.append((TURNOVER_amount,trade_date,stock_no))
            if len(rows)>1000:
                i = i + 1000
                print(i)
                upate_db(rows,conn=conn)
                rows=[]
                # break
        upate_db(rows,conn=conn)

if __name__ == "__main__":
    process() 
