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


def process(dataset_type,field="highN_rate",diff_val_thr=8.0):
    """
    field:highN_rate,lowN_rate,next_high_rate,next_low_rate
    """
    dtmap = {"train":0,"validate":1,"test":2,"predict":3}
    dstype = dtmap.get(dataset_type)
        
    df = pd.read_csv(f"data/btc/all_train_data.csv",sep=";",header=0,dtype={'trade_date':int})
    df = df[df['dataset_type']==dstype]
    df = df.sort_values(by=[field],ascending=False)
    df = df.reset_index(drop=True)
    
    all_pairs = []
    for idx_i,i_row in df.iterrows():
        for idx_j,j_row in df.iterrows():
            if idx_i>idx_j:
                continue
            diff_val = abs(round(i_row[field]-j_row[field],4))
            if diff_val<diff_val_thr:
                continue
            all_pairs.append([i_row['pk_date_btc'],j_row['pk_date_btc'],diff_val])
            
    df = pd.DataFrame(all_pairs,columns=['id_1','id_2','diff'])
    df.to_csv(f"data/btc/pairs_{dataset_type}_{field}.csv",sep=";",index=None)


if __name__ == "__main__":
    process("validate",field="highN_rate",diff_val_thr=10.0)
    process("test",field="highN_rate",diff_val_thr=10.0)
    process("train",field="highN_rate",diff_val_thr=10.0)
        
