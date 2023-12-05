import os
import sys
import time
import random
import json
# import datetime
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import requests
import re 
import logging
from multiprocessing import Pool
from itertools import islice
import sqlite3  
import pickle
from threading import Thread
from common import load_stocks,c_round
from sklearn.metrics.pairwise import cosine_similarity

# cat schema/case2.txt  | awk -F '|' '{print $2$3"|"$3"|"$2"|"$5"|"$6"|"$7}'

conn = sqlite3.connect("file:day_delta/stocks_h20.db?mode=ro", uri=True)

def load_data(trade_date=0,is_predict=True):
    str_fields = "pk_date_stock,stock_no,high_rate,data_json"
    
    df = None 
    if is_predict:
        df = pd.read_csv(f"day_delta.{trade_date}.txt",sep=";",header=None,
                         names=str_fields.split(","),dtype={'stock_no': str})
    else:
        sql = f"select {str_fields} from stock_simple_feathers where trade_date={trade_date}"
        df = pd.read_sql(sql, conn)
    df = df[ (df['stock_no'].str.startswith('688') == False)]
    
    vectors = []
    for idx,row in df.iterrows():
        vectors.append(json.loads(row['data_json'])) #生成时，直接生成字符串，就不需要json.loads()处理了 
        
    return df,vectors 
    

def get_similar_scores(case_file,trade_date=0,is_predict=True):
    pk_date_stocks = "" #样本要有代表性，每天抽取一个最好的？
    with open(f'schema/{case_file}.txt','r') as f:
        pk_date_stocks = ",".join([line.strip().split("|")[0] for line in f])
        
    sql = f"select pk_date_stock,data_json from stock_simple_feathers where pk_date_stock in ({pk_date_stocks})"
    df_samples = pd.read_sql(sql, conn)
    
    vectors_sample = []
    score_columns = []
    for idx,row in df_samples.iterrows():
        score_columns.append("s"+str(row['pk_date_stock']))
        vectors_sample.append(json.loads(row['data_json'])) 
        
    # v = json.loads(df_samples['data_json'][0])
    # print(v)
    # score_columns = [f"s{i}" for i in range(len(vectors_sample))]
    print(score_columns)
     

    df,vectors = load_data(trade_date,is_predict)  
    df["score_cnt"] = 0 
    
    df[score_columns] = cosine_similarity(vectors, vectors_sample) 
    for field in score_columns:
        df['score_cnt'] = df.apply(lambda x: x['score_cnt'] + 1 if x[field]>0.81 else x['score_cnt'], axis=1)
        
    df = df.sort_values(by=['score_cnt'],ascending=False)
    # df = df.drop(['data_json'],axis=1)
    df = df["pk_date_stock,stock_no,high_rate,score_cnt".split(",")] 
    df.to_csv(f"cosie_{case_file}_{trade_date}.txt",sep=";",index=None)
    # print(df.head(10))
    # score>0.81

if __name__ == "__main__":
    get_similar_scores("case_101",20231124,is_predict=False)
    get_similar_scores("case_101",20231204,is_predict=True)
    # get_similar_scores("case_01",20231124,is_predict=False)
    # get_similar_scores("case_01",20231204,is_predict=True)