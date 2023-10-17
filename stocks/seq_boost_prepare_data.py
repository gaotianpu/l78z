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

from seq_model import StockForecastModel,evaluate_ndcg_and_scores
from common import load_trade_dates

SEQUENCE_LENGTH = 20 #序列长度
D_MODEL = 9  #维度 9,17,29


MODEL_FILE = "model_point_sampled.pth" 

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  #苹果的Metal Performance Shaders（MPS）
    if torch.backends.mps.is_available()
    else "cpu"
)

class StockPointDataset(Dataset):
    def __init__(self,datatype="validate",trade_date=None, field="f_high_mean_rate"): 
        dtmap = {"train":0,"validate":1,"test":2}
        self.field = field
        self.conn = sqlite3.connect("file:data/stocks_train_3.db?mode=ro", uri=True)
        dataset_type = dtmap.get(datatype)
        
        if trade_date:
            sql = (
                "select pk_date_stock from stock_for_transfomer where trade_date='%s'"
                % (trade_date)
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
        data = data[:,:9] #9,17,29
        return pk_date_stock, torch.tensor(list_label), torch.tensor(true_score), torch.tensor(high_1), torch.tensor(low_1), torch.tensor(open_rate),  data

def process(models,data_type="train",trade_date=None):
    dataset = StockPointDataset(data_type,trade_date)
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
                
        df_all.to_csv("data/boost_data/boost_%s_%s.txt"%(data_type,trade_date),sep=";",index=False)

def process_by_dates():
    # 加载模型文件
    order_models = "pair_15,list_235,point_5,point_4,pair_11".split(",")
    model_files = order_models + "point_high1,low1.7".split(",")

    models = {}
    for model_name in model_files:
        print(model_name)
        model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
        mfile = "model_v2/StockForecastModel.pth.%s"%(model_name)
        if os.path.isfile(mfile):
            model.load_state_dict(torch.load(mfile)) 
        model.eval()
        models[model_name] = model
        
    conn = sqlite3.connect("file:data/stocks_train_3.db?mode=ro", uri=True)
    trade_dates = load_trade_dates(conn,0)
    for idx,date in enumerate(trade_dates):
        print(idx,date)
        process(models,"train",date)

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
if __name__ == "__main__":
    op_type = sys.argv[1]
    print(op_type)
    if op_type == "prepare":
        process_by_dates()
    if op_type == "min_max":
        get_min_max()
    
                           
                
            
            
            