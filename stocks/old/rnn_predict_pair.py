#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import torch
from torch.utils.data import DataLoader
from rnn_train_point import StockDataset
from rnn_train_pair import PairWiseModel

def run_predict(model_dir="model"):
    dataset = StockDataset("predict")
    _, _, _,trade_date = dataset[0]

    dataloader = DataLoader(dataset, 
        batch_size=64, shuffle=False)

    model = torch.load('%s/rnn_pair_7.pth'%(model_dir))
    model.eval() #预测模式
    with torch.no_grad():
        li = []
        for X, y, stock_no,_ in dataloader:
            # print(trade_date)
            pred_scores = model.predict(X)
            li = li + list(zip(stock_no.tolist(),pred_scores.squeeze().tolist()))
        
        content = "\n".join(['%s,%0.2f' %(str(s[0]).zfill(6),s[1]) for s in li])
        with open("predict/rnn_predict_pair_scores_%d.txt"%(trade_date),'w') as f:
            f.write(content)
        with open("predict/rnn_predict_pair_scores.txt",'w') as f:
            f.write(content)

if __name__ == "__main__":
    model_dir = sys.argv[1] if sys.argv[1] else "model"
    run_predict(model_dir)