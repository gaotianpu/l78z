import os
import sys
from typing import Optional
import numpy as np
import pandas as pd
import json
import sqlite3
import math


df_predict_v1 = pd.read_csv("data/predict/predict_merged.txt",sep=";",header=0,dtype={'stock_no': str})
df_predict_v1 = df_predict_v1.rename(columns={'top3':'top3_v1'})

df_predict_v2 = pd.read_csv("data/predict_v2/predict_merged.txt",sep=";",header=0,dtype={'stock_no': str})
sel_fields = "stock_no,top3,list_dates,point_low1".split(",")
df_predict_v2 = df_predict_v2[sel_fields]
df_predict_v2 = df_predict_v2.rename(columns={'top3':'top3_v2'})

df_predict_v1 = df_predict_v1.merge(df_predict_v2,on="stock_no",how='left')
df_predict_v1["top3"] = df_predict_v1["top3_v1"] + df_predict_v1["top3_v2"]

# df_predict_v1 = df_predict_v1.sort_values(by=["top3","list_dates"],ascending=False) # 
df_predict_v1 = df_predict_v1.sort_values(by=["top3","pair_15"],ascending=False) # 
df_predict_v1.to_csv("data/predict_v2/predict_merged_v1_2.txt",sep=";",index=False) 

# # 只关注预测结果top3=3部分的数据
# df_predict_v2 = df_predict_v2[df_predict_v2['top3']==3]
# df_predict_v2['lowest_rate'] = df_predict_v2['point_low1']