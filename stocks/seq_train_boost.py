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

from common import load_trade_dates
from seq_model import StockForecastModel,StockPointDataset,SEQUENCE_LENGTH,D_MODEL,evaluate_ndcg_and_scores

# SEQUENCE_LENGTH = 20 #序列长度
# D_MODEL = 9  #维度

def model_boost():
    test_data = StockPointDataset(datatype="test",field="f_high_mean_rate")
    dataloader = DataLoader(test_data, batch_size=120) 
    
    df_merged = None
    
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)         
    model_files="point_pair_high,pair_11,pair_15,pair_16,point_4,point_5".split(",")
    for model_name in model_files:
        print(model_name)
        mfile = "model_v2/StockForecastModel.pth.%s"%(model_name)
        if os.path.isfile(mfile):
            model.load_state_dict(torch.load(mfile)) 
        
        model.eval()
        with torch.no_grad():
            all_li = []
            for _batch, (pk_date_stock,true_scores,list_label,data) in enumerate(dataloader): 
                output = model(data.to(device))  
                # 准备计算分档loss，ndcg相关的数据
                if df_merged is None:
                    all_li = all_li + list(zip(pk_date_stock.tolist(), true_scores.tolist(),list_label.tolist(), output.tolist()))
                else:
                    all_li = all_li + list(zip(pk_date_stock.tolist(), output.tolist()))
                # break 
            
            # 模型结果合并
            if df_merged is None:
                df_merged = pd.DataFrame(all_li,columns=["pk_date_stock","true","label",model_name])
                df_merged["trade_date"] = df_merged.apply(lambda x: str(x['pk_date_stock'])[:8] , axis=1)
            else: # 
                df = pd.DataFrame(all_li,columns=["pk_date_stock",model_name])
                df_merged = df_merged.merge(df, on="pk_date_stock",how='left')
            df_merged = df_merged.round({model_name: 4}) 
    
    # df_merged["trade_date"] = df_merged.apply(lambda x: str(x['pk_date_stock'])[:8] , axis=1)
    df_merged.to_csv("data/test_all.txt",sep=";",index=False) 
     
    #
    boost_mean_scores(df_merged,model_files)
    

def boost_mean_scores(df_merged,model_files):
    li_ndcg = []
    date_groups = df_merged.groupby('trade_date')
    for trade_date,data in date_groups:
        count = len(data)
        top3 = int(count/8)  # =count*3/24
        top5 = int(count*5/24) 
                
        for model in model_files:
            # print(model)
            data = data.sort_values(by=[model],ascending=False) #  
            data[model + '_top3'] = [1 if i<top3 else 0 for i in range(count)]
            data[model + '_top5'] = [1 if i<top5 else 0 for i in range(count)]
            
        data['top_3'] = data.apply(lambda x: x[[model+'_top3' for model in model_files]].sum() , axis=1)
        data['top_5'] = data[[ model+'_top5' for model in model_files ]].sum(axis=1)
        
        # data['top_5'] = data.apply(lambda x: x[[model+'_top5' for model in model_files]].sum() , axis=1)
        # data['top_3'] = data[[ model+'_top3' for model in model_files ]].sum(axis=1)
        # data.to_csv("data/tmp.txt",sep=";",index=False) 
        
        y_true_scores = np.expand_dims(data['true'].to_numpy(),axis=0)
        
        y_predict = np.expand_dims(data['top_3'].to_numpy(),axis=0)
        idx = np.argsort(y_predict,axis=1)
        y_true_sorted = np.take(y_true_scores,idx).squeeze()
        ta = y_true_sorted.mean()
        t3_3 = y_true_sorted[-3:].mean()
        t3_5 = y_true_sorted[-5:].mean()
        # print("top_3:",t3_3,t3_5)
        
        
        y_predict = np.expand_dims(data['top_5'].to_numpy(),axis=0)
        idx = np.argsort(y_predict,axis=1)
        y_true_sorted = np.take(y_true_scores,idx).squeeze()
        t5_3 = y_true_sorted[-3:].mean()
        t5_5 = y_true_sorted[-5:].mean()
        # print("top_5:",t5_3,t5_5)
        li_ndcg.append([t3_3,t3_5,t5_3,t5_5])
        # print(trade_date,[t3_3,t3_5,t5_3,t5_5])
        # break 

    mean_scores = [round(v,4) for v in np.mean(li_ndcg,axis=0).tolist()]
    print("mean:t3_3=%s,t3_5=%s,t5_3=%s,t5_5=%s" % tuple(mean_scores) ) 
    
    std_scores = [round(v,4) for v in np.std(li_ndcg,axis=0).tolist()]
    print("std:t3_3=%s,t3_5=%s,t5_3=%s,t5_5=%s" % tuple(std_scores) )  
    
if __name__ == "__main__":
    op_type = sys.argv[1]
    if op_type == "boost": 
        # python seq_train_boost.py boost   
        model_boost()
    
    if op_type == "boost_test": 
        model_files="point_pair_high,pair_11,pair_15,pair_16,point_4,point_5".split(",")
        # python seq_train_boost.py boost_test   
        df_merged = pd.read_csv("data/test_all.txt",sep=";",header=0) 
        df_merged["trade_date"] = df_merged.apply(lambda x: str(x['pk_date_stock'])[:8] , axis=1)
        boost_mean_scores(df_merged,model_files)