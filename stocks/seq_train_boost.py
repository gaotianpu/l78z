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
import random

from common import load_trade_dates
from seq_model import StockForecastModel,StockPointDataset,SEQUENCE_LENGTH,D_MODEL,device,evaluate_ndcg_and_scores

# SEQUENCE_LENGTH = 20 #序列长度
# D_MODEL = 9  #维度


def model_boost():
    test_data = StockPointDataset(datatype="test",field="f_high_mean_rate")
    dataloader = DataLoader(test_data, batch_size=120) 
    
    df_merged = None
    
    order_models = "pair_15,pair_16,point_4,point_5".split(",")
    model_files = order_models + "point_high1,low1.7".split(",")
    
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
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
                # df_merged["stock_no"] = df_merged.apply(lambda x: str(x['pk_date_stock'])[8:] , axis=1)
            else: # 
                df = pd.DataFrame(all_li,columns=["pk_date_stock",model_name])
                df_merged = df_merged.merge(df, on="pk_date_stock",how='left')
            
            # 排序，标示出top3,top5
            if model_name in order_models:
                df_merged = df_merged.sort_values(by=["trade_date",model_name],ascending=False)
                df_merged[model_name + '_top3'] = 0
                df_merged[model_name + '_top5'] = 0
                
                last_date = 0
                topN = 0
                date_len = 0 
                for idx, row in df_merged.iterrows():
                    if last_date != row['trade_date']:
                        last_date = row['trade_date']
                        date_len = len(df_merged[df_merged['trade_date']==last_date] ) 
                        topN = 0
                        
                    if topN < int(date_len/8): # date_len*3/24
                        df_merged.loc[idx,model_name + '_top3'] = 1
                    if topN < int(date_len*5/24):
                        df_merged.loc[idx,model_name + '_top5'] = 1
                    
                    topN = topN + 1 
                
            df_merged = df_merged.round({model_name: 4}) 
    
    # df_predict['top_3'] = df_predict.apply(lambda x: x[[model+'_top3' for model in order_models]].sum() , axis=1)
    df_merged['top_3'] = df_merged[[ model_name+'_top3' for model_name in order_models ]].sum(axis=1)
    df_merged['top_5'] = df_merged[[ model_name+'_top5' for model_name in order_models ]].sum(axis=1) 
    
    df_prices = pd.read_csv("data/test_stock_raw_daily_3.txt",sep=";",header=0)
    df_merged = df_merged.merge(df_prices, on="pk_date_stock",how='left') 
        
    df_static =  pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0)
    df_merged = df_merged.merge(df_static, on="stock_no",how='left')
    
    df_merged.to_csv("data/test_all.txt",sep=";",index=False) 
    #
    boost_mean_scores(df_merged,None)


def boost_mean_scores(df_merged,order_models=None):
    li_ndcg = []
    date_groups = df_merged.groupby('trade_date_x')
    for trade_date,data in date_groups:
        if order_models:
            count = len(data)
            top3 = int(count/8)  # =count*3/24
            top5 = int(count*5/24) 
                    
            for model in order_models:
                data = data.sort_values(by=[model],ascending=False) #  
                data[model + '_top3'] = [1 if i<top3 else 0 for i in range(count)]
                data[model + '_top5'] = [1 if i<top5 else 0 for i in range(count)]
            
                # ? KeyError: "None of [Index(['pair_15_top3', 'pair_16_top3', 'point_4_top3', 'point_5_top3'], dtype='object')] are in the [index]"
                # df_merged[model + '_top3'] = 0
                # df_merged.loc[data.index, model + '_top3'] = [1 if i<top3 else 0 for i in range(count)]
            
            data['top_3'] = data[[ model+'_top3' for model in order_models ]].sum(axis=1)    
            # data['top_3'] = data.apply(lambda x: x[[model+'_top3' for model in order_models]].sum() , axis=1)
            data['top_5'] = data[[ model+'_top5' for model in order_models ]].sum(axis=1)
            # data['top_5'] = data.apply(lambda x: x[[model+'_top5' for model in order_models]].sum() , axis=1)
            
        # df_merged.to_csv("data/tmp.txt",sep=";",index=False) 
        
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

    str_models = ""
    if order_models:
        str_models = ",".join(order_models)
        
    mean_scores = [round(v,4) for v in np.mean(li_ndcg,axis=0).tolist()]
    print(str_models + " mean:t3_3=%s,t3_5=%s,t5_3=%s,t5_5=%s" % tuple(mean_scores) ) 
    
    std_scores = [round(v,4) for v in np.std(li_ndcg,axis=0).tolist()]
    print(str_models + " std:t3_3=%s,t3_5=%s,t5_3=%s,t5_5=%s\n" % tuple(std_scores) )
    return mean_scores[0]

def c_round(x):
    return round(x,4)

def buy_sell_model(df,module_name="all"):
    # 成功买入
    low_std = 0.013194
    high_std = 0.018595
    df['can_buy_0'] = df.apply(lambda x: 1 if (x['low1.7'] - low_std) > x['low_rate'] else 0  , axis=1)
    df['can_buy_1'] = df.apply(lambda x: 1 if (x['low1.7']) > x['low_rate'] else 0  , axis=1)
    df['can_buy_2'] = df.apply(lambda x: 1 if (x['low1.7'] + low_std) > x['low_rate'] else 0  , axis=1)
    
    df['can_buy_3'] = df.apply(lambda x: 1 if (x['low_rate_25%']) > x['low_rate'] else 0  , axis=1)
    df['can_buy_4'] = df.apply(lambda x: 1 if (x['low_rate_50%']) > x['low_rate'] else 0  , axis=1)
    df['can_buy_5'] = df.apply(lambda x: 1 if (x['low_rate_75%']) > x['low_rate'] else 0  , axis=1)
    
    # 成功卖出
    df['can_sell_0'] = df.apply(lambda x: 1 if (x['point_high1'] - high_std) < x['high_rate'] else 0  , axis=1)
    df['can_sell_1'] = df.apply(lambda x: 1 if (x['point_high1']) < x['high_rate'] else 0  , axis=1)
    df['can_sell_2'] = df.apply(lambda x: 1 if (x['point_high1'] + high_std) < x['high_rate'] else 0  , axis=1)
    
    df['can_sell_3'] = df.apply(lambda x: 1 if (x['high_rate_25%']) < x['high_rate'] else 0  , axis=1)
    df['can_sell_4'] = df.apply(lambda x: 1 if (x['high_rate_50%']) < x['high_rate'] else 0  , axis=1)
    df['can_sell_5'] = df.apply(lambda x: 1 if (x['high_rate_75%']) < x['high_rate'] else 0  , axis=1)
    # low_rate > low1.7
    
    li_ = []
    total = len(df)
    for i in range(6):
        li_.append( c_round(len(df[df['can_buy_'+str(i)]==1])/total)) 
    
    for i in range(6):
        li_.append( c_round(len(df[df['can_sell_'+str(i)]==1])/total)) 
    
    # buy_success:0.0854,0.356,0.9308,0.2808,0.5117,0.7158; 
    # sel_success:0.9619,0.4845,0.1145,0.7391,0.5213,0.2724.
    print(module_name + " buy_success,%s,%s,%s,%s,%s,%s, sel_success,%s,%s,%s,%s,%s,%s" % tuple(li_) )  

def buy_sell():     
    df =  pd.read_csv("data/test_all.txt",sep=";",header=0)  
    
    # df = pd.read_csv("data/test_stock_raw_daily_3.txt",sep=";",header=0)
    # # ,dtype={'stock_no': str}
    
    # df_static =  pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0)   
    # df = df.merge(df_static, on="stock_no",how='left')
    
    # df_predict =  pd.read_csv("data/test_all.txt",sep=";",header=0)  
    # df = df.merge(df_predict, on="pk_date_stock",how='left')
    
    order_models = "pair_15,pair_16,point_4,point_5".split(",")
    for model_name in order_models:
        df_top3 = df[df[model_name+"_top3"]==1]
        buy_sell_model(df_top3.copy(),model_name+"_top3")
        df_top5 = df[df[model_name+"_top5"]==1]
        buy_sell_model(df_top5.copy(),model_name+"_top5")
    
    for i in range(4):
        df_top3 = df[df["top_3"]>i]
        buy_sell_model(df_top3.copy(),"all_top3_%s"%(i))
        df_top5 = df[df["top_5"]>i]
        buy_sell_model(df_top5.copy(),"all_top5_%s"%(i))
        
    buy_sell_model(df,"all")
    
    df.to_csv("data/tmp.txt",sep=";",index=False) 

import random
def boost_mean_scores_t(df_merged, m):
    return random.random()
    
def best_boost_mean_scores():    
    # python seq_train_boost.py boost_test   
    df_merged = pd.read_csv("data/test_all.txt",sep=";",header=0) 
    # df_merged["trade_date"] = df_merged.apply(lambda x: str(x['pk_date_stock'])[:8] , axis=1)
    order_models="point_pair_high,pair_11,pair_15,pair_16,point_4,point_5,point_high1".split(",")
    
    best_models = []
    current_best_score = 0
    for i in range(7):
        li_ = []
        for model in order_models:
            if model in best_models:
                continue  
            
            mli = best_models + [model]
            score = boost_mean_scores_t(df_merged, mli)
            li_.append( [mli,score] )
        
        best = max(li_, key = lambda x: x[1])
        if best[1] > current_best_score:
            best_models = best[0]
            current_best_score = best[1]
        else :
            print("???", best[0])
    
    print("best_models:",best_models)
    
                
if __name__ == "__main__":
    op_type = sys.argv[1]
    if op_type == "boost": 
        # python seq_train_boost.py boost   
        model_boost()
    
    if op_type == "buy_sell":
        #python seq_train_boost.py buy_sell 
        buy_sell()
    
    if op_type == "best_boost": 
        best_boost_mean_scores()
        
        # df = pd.read_csv("data/test_stock_raw_daily_3.txt",sep=";",header=0)
        # # ,dtype={'stock_no': str}
        
        # df_static =  pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0)   
        # df = df.merge(df_static, on="stock_no",how='left')
        
        # df =  pd.read_csv("data/test_all.txt",sep=";",header=0)  
        # # df = df.merge(df_predict, on="pk_date_stock",how='left')
        
        # # df_merged = pd.read_csv("data/test_all.txt",sep=";",header=0) 
        # # df_merged["trade_date"] = df_merged.apply(lambda x: str(x['pk_date_stock'])[:8] , axis=1)
        # order_models="pair_15,pair_16,point_4,point_5".split(",")
        # boost_mean_scores(df,order_models)
    
    if op_type == "tmp":
        df_predict =  pd.read_csv("data/test_all.txt",sep=";",header=0)  
        order_models="pair_15,pair_16,point_4,point_5".split(",")
        for model in order_models:
            df_predict = df_predict.sort_values(by=["trade_date",model],ascending=False)
            df_predict[model + '_top3'] = 0
            df_predict[model + '_top5'] = 0
            
            last_date = 0
            topN = 0
            date_len = 0 
            for idx, row in df_predict.iterrows():
                if last_date != row['trade_date']:
                    last_date = row['trade_date']
                    topN = 0
                    date_len = len(df_predict[df_predict['trade_date']==last_date] )   
                    
                if topN < 3:
                    df_predict.loc[idx,model + '_top3'] = 1
                if topN < 5:
                    df_predict.loc[idx,model + '_top5'] = 1
                
                topN = topN + 1 
                
        # df_predict['top_3'] = df_predict.apply(lambda x: x[[model+'_top3' for model in order_models]].sum() , axis=1)
        df_predict['top_3'] = df_predict[[ model+'_top3' for model in order_models ]].sum(axis=1)
        df_predict['top_5'] = df_predict[[ model+'_top5' for model in order_models ]].sum(axis=1)
        
        df_predict.to_csv("data/tmp.txt",sep=";",index=False) 
        
            