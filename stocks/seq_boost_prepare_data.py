import os
import sys
from typing import Optional
import numpy as np
import pandas as pd
import json
import sqlite3
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import ndcg_score

from seq_model_v2 import StockForecastModel,evaluate_ndcg_and_scores,SEQUENCE_LENGTH,D_MODEL,device
from common import load_trade_dates,load_stocks

class StockPointDataset(Dataset):
    def __init__(self, datatype="validate", split_by="dates",split_param=None): 
        dtmap = {"train":0,"validate":1,"test":2}
        # self.field = field #, field="f_high_mean_rate"
        self.conn = sqlite3.connect("file:data/stocks_train_3.db?mode=ro", uri=True)
        dataset_type = dtmap.get(datatype)
        
        if split_by=="dates":
            sql = (
                "select pk_date_stock from stock_for_transfomer where trade_date='%s'"
                % (split_param)
            )
            self.df = pd.read_sql(sql, self.conn)
        elif split_by=="stocks":
            sql = (
                "select pk_date_stock from stock_for_transfomer where stock_no='%s'"
                % (split_param)
            )
            self.df = pd.read_sql(sql, self.conn)
        else:
            self.df = pd.read_csv('data/seq_train_%s.txt' % (dataset_type),header=None)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pk_date_stock = self.df.iloc[idx][0] 
        sql = "select * from stock_for_transfomer where pk_date_stock=%s" % (pk_date_stock)
        df_item = pd.read_sql(sql, self.conn) 
        
        data_json = json.loads(df_item.iloc[0]['data_json']) #.replace("'",'"'))
        
        true_score = data_json.get("f_high_mean_rate")
        high_1 = data_json.get("next_high_rate")
        low_1 = data_json.get("next_low_rate")
        open_rate = data_json.get("next_open_rate")
        
        list_label = df_item.iloc[0]['list_label'] 
        
        data = torch.tensor(data_json["past_days"])
        # data = data[:,:9] #9,17,29
        return pk_date_stock, torch.tensor(list_label), torch.tensor(true_score), torch.tensor(high_1), torch.tensor(low_1), torch.tensor(open_rate),  data

def process(models,data_type="train",split_by="dates",split_param=None):
    df_file = "data/boost_%s_%s/boost_%s.txt"%(split_by,data_type,split_param)
    if os.path.exists(df_file):
        print(df_file, "is exist")
        return 
    
    dataset = StockPointDataset(data_type, split_by, split_param)
    dataloader = DataLoader(dataset, batch_size=128) 
    
    # 先把每个模型的得分，需要用到的各种数据整理一遍
    with torch.no_grad():
        df_all = None 
        for _batch, (pk_date_stock, list_label, true_score, high_1, low_1, open_rate,  data) in enumerate(dataloader):
            # print(_batch)
            df = pd.DataFrame(pk_date_stock.tolist(),columns = ["pk_date_stock"])
            df["trade_date"] = df.apply(lambda x: int(str(x['pk_date_stock'])[:8]) , axis=1)
            df["stock_no"] = df.apply(lambda x: str(x['pk_date_stock'])[8:] , axis=1)
            df["list_label"] = list_label.tolist()
            df["true_score"] = true_score.tolist()
            df["true_high_1"] = high_1.tolist()
            df["true_low_1"] = low_1.tolist()
            df["true_open_rate"] = open_rate.tolist() 
            
            for model_name,model in  models.items(): 
                output = model(data.to(device)) 
                df[model_name] = output.tolist() 
                
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all,df],axis=0)
        
        if df_all is not None:        
            df_all.to_csv(df_file,sep=";",index=False)

def process_all(by="dates"): # by_dates():
    # 加载模型文件
    # order_models = "pair_15,list_235,point_5,point_4,pair_11".split(",")
    # model_files = order_models + "point_high1,low1.7".split(",")
    model_files = ["pair_stocks"]
    
    models = {}
    for model_name in model_files:
        print(model_name)
        model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
        mfile = "model_v3/model_%s.pth"%(model_name)
        if os.path.isfile(mfile):
            model.load_state_dict(torch.load(mfile)) 
        model.eval()
        models[model_name] = model
        
    if by == "dates":
        conn = sqlite3.connect("file:data/stocks_train_3.db?mode=ro", uri=True)
        trade_dates = load_trade_dates(conn,0)
        for idx,date in enumerate(trade_dates):
            print(idx,date)
            process(models,"train",by, date)
    elif by == "stocks":
        conn = sqlite3.connect("file:newdb/stocks.db?mode=ro", uri=True)
        stocks = load_stocks(conn)
        for idx,stock in enumerate(stocks):
            print(idx,stock[0])
            process(models,"train",by,stock[0]) 
    else:
        pass 
        

def get_min_max():
    conn = sqlite3.connect("file:data/stocks_train_4.db?mode=ro", uri=True)
    fields = "pair_15,list_235,point_5,point_4,pair_11,point_high1,low1,true_open_rate".split(",")
    sql_min_max_parts = ",".join(["min(%s),max(%s)"%(field,field) for field in fields])
    sql = "select %s from stock_for_boost_v2" %(sql_min_max_parts)
    df = pd.read_sql(sql, conn)
    d = {}
    for i,field in enumerate(fields):
        d[field+"_min"] = df.iloc[0][i*2] 
        d[field+"_max"] = df.iloc[0][i*2 + 1]
    print(d) 
    

# mv seq_train_boost_v2.py seq_boost_prepare_data.py
# python seq_boost_prepare_data.py stocks > log/seq_boost_prepare_data_stocks.log
if __name__ == "__main__":
    op_type = sys.argv[1]
    print(op_type)
    if op_type == "dates":
        process_all(by="dates")
    if op_type == "stocks":
        process_all(by="stocks")
    if op_type == "min_max":
        get_min_max()
    
                           
                
            
            
            