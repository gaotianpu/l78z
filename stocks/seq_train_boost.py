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

from common import c_round
from seq_model_v2 import StockForecastModel,StockPointDataset,SEQUENCE_LENGTH,D_MODEL,device,evaluate_ndcg_and_scores

# SEQUENCE_LENGTH = 20 #序列长度
# D_MODEL = 9  #维度

true_test_file = "data/model_test_2/true_test.txt"

def get_model_results(model_name,dataloader):
    # 从已经存在的文件中加载
    model_results_file = "data/model_test_2/%s.txt" % (model_name)
    if os.path.isfile(model_results_file):
        df = pd.read_csv(model_results_file,sep=";",header=0)
        return df
    
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device) 
    # mfile = "model_v2/StockForecastModel.pth.%s"%(model_name)
    mfile = "model_v3/model_%s.pth"%(model_name)
    if os.path.isfile(mfile):
        model.load_state_dict(torch.load(mfile)) 
    
    model.eval()
    with torch.no_grad():
        all_li = []
        for _batch, (pk_date_stock,true_scores,list_label,data) in enumerate(dataloader): 
            output = model(data.to(device))  
            all_li = all_li + list(zip(pk_date_stock.tolist(), output.tolist()))             
        
        df = pd.DataFrame(all_li,columns=["pk_date_stock",model_name])
        df = df.round({model_name: 4}) 
        df["trade_date"] = df.apply(lambda x: str(x['pk_date_stock'])[:8] , axis=1)
        df = df.sort_values(by=["trade_date",model_name],ascending=False)
        
        df[model_name + '_top3'] = 0
        df[model_name + '_top5'] = 0
        
        last_date = 0
        topN = 0
        date_len = 0 
        for idx, row in df.iterrows():
            if last_date != row['trade_date']:
                last_date = row['trade_date']
                date_len = len(df[df['trade_date']==last_date] ) 
                topN = 0
                
            if topN < int(date_len/8): # date_len*3/24
                df.loc[idx,model_name + '_top3'] = 1
            if topN < int(date_len*5/24):
                df.loc[idx,model_name + '_top5'] = 1
            
            topN = topN + 1
        
        # 基于trade_date,model_scores排序，设定topN后，该列在后续合并时容易重复，删除之
        df = df.drop(['trade_date'],axis=1) 
        df.to_csv(model_results_file,sep=";",index=False)
        return df
        
def model_boost():
    test_data = StockPointDataset(datatype="test",field="f_high_mean_rate")
    dataloader = DataLoader(test_data, batch_size=120)
    
    # df = get_model_results("list_235",dataloader)
    # return 
    
    # 非模型相关的数据
    df_merged = None
    true_test_file = "data/model_test_2/true_test.txt"
    if os.path.isfile(true_test_file): #缓存
        df_merged = pd.read_csv(true_test_file,sep=";",header=0)
    else:
        all_li = []
        for _batch, (pk_date_stock,true_scores,list_label,data) in enumerate(dataloader):  
            all_li = all_li + list(zip(pk_date_stock.tolist(), true_scores.tolist(),list_label.tolist()))
        df_merged = pd.DataFrame(all_li,columns=["pk_date_stock","true","label"])
        df_merged["trade_date"] = df_merged.apply(lambda x: str(x['pk_date_stock'])[:8] , axis=1)
        df_merged.to_csv(true_test_file,sep=";",index=False)
    
    # 模型预测相关数据
    # order_models = "pair_15,pair_16,point_4,point_5".split(",")
    # model_files = order_models + "point_high1,low1.7".split(",")
    
    # order_models = "point,point2pair_dates,pair_dates,list_dates,pair_dates_stocks".split(",")
    order_models = "list_dates,point,point2pair_dates".split(",")
    model_files = order_models + "point_low,point_low1".split(",") #point_high1,
    
    for model_name in model_files:
        print(model_name)
        df = get_model_results(model_name,dataloader)
        df_merged = df_merged.merge(df, on="pk_date_stock",how='left')
    
    # 模型boost，多个模型1/0,简单相加
    # df_predict['top_3'] = df_predict.apply(lambda x: x[[model+'_top3' for model in order_models]].sum() , axis=1)
    df_merged['top_3'] = df_merged[[ model_name+'_top3' for model_name in order_models ]].sum(axis=1)
    df_merged['top_5'] = df_merged[[ model_name+'_top5' for model_name in order_models ]].sum(axis=1) 
    
    df_merged = df_merged.sort_values(by=["trade_date","top_3"],ascending=False)
    df_merged.to_csv("data/model_test_2/test_merged.txt",sep=";",index=False) 
    
    df_prices = pd.read_csv("data/test_stock_raw_daily_3.txt",sep=";",header=0)
    # 列重命名
    df_prices.rename(columns={'trade_date':'trade_date_next'},inplace=True) 
    df_merged = df_merged.merge(df_prices, on="pk_date_stock",how='left') 
        
    df_static =  pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0)
    df_merged = df_merged.merge(df_static, on="stock_no",how='left')
    
    df_merged.to_csv("data/model_test_2/test_all.txt",sep=";",index=False) 
    #
    boost_mean_scores(df_merged,order_models)


def boost_mean_scores(df_merged,order_models=None):
    li_ndcg = []
    date_groups = df_merged.groupby('trade_date')
    str_order_models = "_".join(order_models)
    for trade_date,data in date_groups:
        if order_models:
            data['top_3'] = data[[ model+'_top3' for model in order_models ]].sum(axis=1) 
            data['top_5'] = data[[ model+'_top5' for model in order_models ]].sum(axis=1)
            # data['top_3'] = data.apply(lambda x: x[[model+'_top3' for model in order_models]].sum() , axis=1)
            # data['top_5'] = data.apply(lambda x: x[[model+'_top5' for model in order_models]].sum() , axis=1)
            
            # data.to_csv("data/model_test_2/ms_t_%s_%s.txt"%(str_order_models,trade_date),sep=";",index=False) 
            # break
            # print(data)
            # count = len(data)
            # top3 = int(count/8)  # =count*3/24
            # top5 = int(count*5/24) 
                    
            # for model in order_models:
            #     data = data.sort_values(by=[model],ascending=False) #  
            #     data[model + '_top3'] = [1 if i<top3 else 0 for i in range(count)]
            #     data[model + '_top5'] = [1 if i<top5 else 0 for i in range(count)]
            
                # ? KeyError: "None of [Index(['pair_15_top3', 'pair_16_top3', 'point_4_top3', 'point_5_top3'], dtype='object')] are in the [index]"
                # df_merged[model + '_top3'] = 0
                # df_merged.loc[data.index, model + '_top3'] = [1 if i<top3 else 0 for i in range(count)]
            
            
            
        # df_merged.to_csv("data/tmp.txt",sep=";",index=False) 
        y_true_labels = np.expand_dims(data['label'].to_numpy(),axis=0) 
        y_true_scores = np.expand_dims(data['true'].to_numpy(),axis=0)
        
        y_predict = np.expand_dims(data['top_3'].to_numpy(),axis=0)
        
        ndcg_3 = round(ndcg_score(y_true_labels,y_predict),3)
        ndcg_3_3 = round(ndcg_score(y_true_labels,y_predict,k=3),3)
        ndcg_3_5 = round(ndcg_score(y_true_labels,y_predict,k=3),5)
        
        idx = np.argsort(y_predict,axis=1)
        y_true_sorted = np.take(y_true_scores,idx).squeeze()
        ta = y_true_sorted.mean()
        t3_3 = y_true_sorted[-3:].mean()
        t3_5 = y_true_sorted[-5:].mean()
        # print("top_3:",t3_3,t3_5)
        # print(y_true_sorted[-3:])
        
        y_predict = np.expand_dims(data['top_5'].to_numpy(),axis=0)
        ndcg_5 = round(ndcg_score(y_true_labels,y_predict),3)
        ndcg_5_3 = round(ndcg_score(y_true_labels,y_predict,k=3),3)
        ndcg_5_5 = round(ndcg_score(y_true_labels,y_predict,k=3),5)
        
        idx = np.argsort(y_predict,axis=1)
        y_true_sorted = np.take(y_true_scores,idx).squeeze()
        t5_3 = y_true_sorted[-3:].mean()
        t5_5 = y_true_sorted[-5:].mean()
        # print(y_true_sorted[-5:])
        # print("top_5:",t5_3,t5_5)
        li_ndcg.append([t3_3,t3_5,t5_3,t5_5,ndcg_3_3,ndcg_3_5,ndcg_5_3,ndcg_5_5])
        # print(trade_date,[t3_3,t3_5,t5_3,t5_5])
        # break 

    str_models = ""
    if order_models:
        str_models = ",".join(order_models)
        
    mean_scores = [c_round(v) for v in np.mean(li_ndcg,axis=0).tolist()]
    print(str_models + " mean:t3_3=%s,t3_5=%s,t5_3=%s,t5_5=%s,n3_3=%s,n3_5=%s,n5_3=%s,n5_5=%s" % tuple(mean_scores) ) 
    
    std_scores = [c_round(v) for v in np.std(li_ndcg,axis=0).tolist()]
    print(str_models + " std:t3_3=%s,t3_5=%s,t5_3=%s,t5_5=%s,n3_3=%s,n3_5=%s,n5_3=%s,n5_5=%s\n" % tuple(std_scores) )
    return mean_scores[4] #ndcg_3_3



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

    
def best_boost_mean_scores(): 
    # 采用动态规划思路，先找到单个最优模型，再基于这单个最优模型，寻找2个最优组，依次3个，4个，当数值不再优化，退出   
    # python seq_train_boost.py boost_test 
    df_merged = pd.read_csv("data/model_test_2/true_test.txt",sep=";",header=0)
    
    # order_models="point_pair_high,pair_11,pair_15,pair_16,point_4,point_5,point_high1,list_235".split(",")
    order_models="point,point2pair_dates,pair_dates,list_dates,pair_dates_stocks".split(",")
    
    # current best is less than last ['point_5', 'point_high1'] ['point_5', 'point_high1', 'pair_15']
    # best_models: ['point_5', 'point_high1']
        
    #pair_15,pair_16,point_4,point_5
        
    best_models = [] #"pair_15,point_4,point_5,pair_11".split(",") #
    current_best_score = 0
    for i in range(7):
        li_ = []
        for model_name in order_models:
            if model_name in best_models:
                continue
            
            mli = best_models + [model_name]
            # print(mli)
            
            df = df_merged.copy()
            for m in mli:
                df_t = pd.read_csv("data/model_test_2/%s.txt" % (m),sep=";",header=0)
                df = df.merge(df_t, on="pk_date_stock",how='left') 
            
            # df.to_csv("data/tmp.txt",sep=";",index=False) 
            
            score = boost_mean_scores(df, mli)
            li_.append( [mli,score] )
        
        best = max(li_, key = lambda x: x[1])
        if best[1] > current_best_score:
            best_models = best[0]
            current_best_score = best[1]
        else :
            print("current best is less than last", best_models, best[0])
            break
            
        # pair_15,point_4,point_5,pair_11
        print("best_models:",best_models)
        # break
    
    print("best_models:",best_models)
    
                
if __name__ == "__main__":
    op_type = sys.argv[1]
    if op_type == "boost": 
        # python seq_train_boost.py boost 
        # data/model_test_2/  
        model_boost()
    
    if op_type == "best_boost": 
        best_boost_mean_scores() 
        
    if op_type == "buy_sell":
        #python seq_train_boost.py buy_sell 
        buy_sell()
    
    
        
        
    
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
        
            