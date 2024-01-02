#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import random
import json
import re
import requests
import pandas as pd 
import datetime
# from datetime import datetime, timezone, timedelta
import logging
from multiprocessing import Pool
from itertools import islice
import sqlite3

conn = sqlite3.connect("file:data/btc_train.db?mode=ro", uri=True)

minmax_threshold={'highN_rate': [-13.4735, 19.4128], 'lowN_rate': [-17.3754, 14.3518], 'high1_rate': [-3.4525, 7.9531], 'low1_rate': [-8.2235, 3.7846]}

def to_csv(df,field,dataset_type=0):
    tdf = df[df['dataset_type']==dataset_type]
    tdf = tdf[['pk_date_btc',f'{field}_rate','label']]
    tdf.to_csv(f'data/cls3_{dataset_type}_{field}.csv',sep=";",index=None)
    
def process(field='highN'):
    thr = minmax_threshold.get(f'{field}_rate')
    min_val,max_val = thr[0],thr[1]
    sql_thr = f" {field}_rate>{min_val} and {field}_rate<{max_val} "
    # dataset_type=0 and 
    sql = f"select pk_date_btc,{field}_rate,dataset_type,0 as label from train_data where {sql_thr} order by {field}_rate desc;"
    df = pd.read_sql(sql,conn)
    
    cnt_3 = int(len(df)/3)
    total_cnt = len(df)
    df['label'] = [2 if i < cnt_3 else 1 if i<cnt_3*2 else 0 for i in range(total_cnt)]
    
    to_csv(df,field,0)
    to_csv(df,field,1)
    to_csv(df,field,2)


if __name__ == "__main__":
    process("highN")