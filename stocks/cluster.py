#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans

def load_stocks_price():
    d = {}
    with open("20220602.txt", 'r') as f:
        for i, line in enumerate(f):
            fields = line.strip().split(',')
            d[fields[0]] = fields[3]
    return d 

def load_stocks_statics(stocks):
    with open("data/statis.txt", 'r') as f:
        for i, line in enumerate(f):
            fields = line.strip().split(',')
            stock_no = fields[0]
            current_price = float(stocks.get(stock_no,-1) )
            if current_price>0:
                # print('current_price:',current_price)
                # print('staic:',line.strip()) 
                print(stock_no+","+",".join([str(round(float(field)/current_price,2)) for ii,field in enumerate(fields) if ii not in [0,1,9,17,25]]))

def process_cluster():
    data = np.loadtxt(open("out_cluster.txt", "r"), delimiter=",")
    stocks = data[:,0]
    features = data[:10,1:]
    # print(data.shape)   
    # print(stocks) 
    # print(features)  

    kmeans = KMeans(n_clusters=2, random_state=0).fit(features) 
    # print(kmeans.labels_)    
            
# python cluster.py > out_cluster.txt 
if __name__ == "__main__": 
    # load_stocks_statics(load_stocks_price())
    process_cluster()
