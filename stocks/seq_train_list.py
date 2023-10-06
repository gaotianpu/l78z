#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from typing import Optional
import pandas as pd
import numpy as np
import sqlite3
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import ndcg_score

from seq_model import StockForecastModel,StockPointDataset,evaluate_ndcg_and_scores,SEQUENCE_LENGTH,D_MODEL

from pytorchltr.loss import PairwiseHingeLoss

# SEQUENCE_LENGTH = 20 #序列长度
# D_MODEL = 9  #维度
MODEL_FILE = "StockForecastModel.list.pth"

# https://blog.csdn.net/qq_36478718/article/details/122598406
# ListNet ・ ListMLE ・ RankCosine ・ LambdaRank ・ ApproxNDCG ・ WassRank ・ STListNet ・ LambdaLoss

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  #苹果的Metal Performance Shaders（MPS）
    if torch.backends.mps.is_available()
    else "cpu"
)

class StockListDataset(Dataset):
    def __init__(self, datatype="train", field="f_high_mean_rate"):
        assert datatype in ("train", "validate", "test")
        self.df = pd.read_csv("list/random_%s.txt" % (datatype), sep=";", header=None)
        self.conn = sqlite3.connect("file:data/stocks_train.db?mode=ro", uri=True)
        self.field = field  # 基于哪个预测值做比较

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        str_pk_date_stock = self.df.iloc[idx][0] #.values.tolist()
        sql = (
            "select pk_date_stock,list_label,data_json from stock_for_transfomer where pk_date_stock in (%s)"
            % (str_pk_date_stock)
        )
        df_list = pd.read_sql(sql, self.conn)
        li_labels = []
        li_data = []
        for idx,row in df_list.iterrows():
            li_labels.append(float(row['list_label']))
            
            data_json = json.loads(row['data_json'])
            li_data.append(data_json['past_days']) 

        return torch.tensor(li_labels),torch.tensor(li_data)
        
# https://pytorchltr.readthedocs.io/en/stable/getting-started.html
# https://github.com/rjagerman/pytorchltr

# https://zhuanlan.zhihu.com/p/148262580

# https://www.cnblogs.com/bentuwuying/p/6690836.html

class ListMLELoss(nn.Module):
    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, k=None
    ) -> torch.Tensor:
        # y_pred : batch x n_items
        # y_true : batch x n_items 
        if k is not None:
            sublist_indices = (y_pred.shape[1] * torch.rand(size=k)).long()
            y_pred = y_pred[:, sublist_indices] 
            y_true = y_true[:, sublist_indices] 
    
        _, indices = y_true.sort(descending=True, dim=-1)
        pred_sorted_by_true = y_pred.gather(dim=1, index=indices)
        cumsums = pred_sorted_by_true.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        listmle_loss = torch.log(cumsums + 1e-10) - pred_sorted_by_true
        return listmle_loss.sum(dim=1).mean() 
    

# 4. train 函数
def train(dataloader, model, loss_fn, optimizer,epoch): 
    model.train() #训练模式
    
    size = len(dataloader.dataset) 
    
    total_loss = 0.0 
    for batch, (labels,data) in enumerate(dataloader):
        predict_scores = model(torch.squeeze(data).to(device)) 
        
        loss = loss_fn(torch.unsqueeze(predict_scores,0), labels) #8*8, 只关注头部1/8？  , (1,8) 
        
        # Back propagation 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
        
        total_loss = total_loss + loss.item()
        
        if batch % 512 == 0:
            avg_loss = total_loss / (batch + 1) 
            loss, current = loss.item(), (batch + 1) * 1 # len(choose)
            print(f"loss: {loss:>7f} , avg_loss: {avg_loss:>7f}  [{epoch:>5d}  {current:>5d}/{size:>5d}]")
                 
        cp_save_n = 512 #cp, checkpoint
        if batch % cp_save_n == 0:
            cp_idx = int(batch / cp_save_n)
            cp_idx_mod = cp_idx % 23
            torch.save(model.state_dict(), "%s.%s.%s" % (MODEL_FILE,epoch,cp_idx_mod) )
            
    torch.save(model.state_dict(), "%s.%s" % (MODEL_FILE,epoch))

# 5. vaildate/test 函数
def test(dataloader, model): 
    model.eval()
    with torch.no_grad():
        all_ret = []
        for _batch, (pk_date_stock,true_scores,list_label,data) in enumerate(dataloader): 
            output = model(data.to(device)) 
            # 准备计算分档loss，ndcg相关的数据
            ret = list(zip(pk_date_stock.tolist(), output.tolist(),true_scores.tolist(),list_label.tolist()))
            all_ret = all_ret + ret   
    
    # 计算ndcg情况
    df = pd.DataFrame(all_ret,columns=["pk_date_stock","predict","true","label"])
    evaluate_ndcg_and_scores(df)
   

def training(field="f_high_mean_rate"):
    # 初始化
    train_data = StockListDataset(datatype="train",field=field)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    # a = next(iter(train_dataloader))
    # print(choose.shape,reject.shape)

    # vali_data = StockPointDataset(datatype="validate",field=field)
    # vali_dataloader = DataLoader(vali_data, batch_size=128)  
    
    test_data = StockPointDataset(datatype="test",field=field)
    test_dataloader = DataLoader(test_data, batch_size=128)  
    
    criterion = ListMLELoss() #
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    
    learning_rate = 0.0000001 #0.00001 #0.000001  #0.0000001  
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=learning_rate, betas=(0.9,0.98), 
                                eps=1e-08) #定义最优化算法

    if os.path.isfile(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE)) 
        print("load success")

    epochs = 5
    start = 0
    for t in range(epochs):
        real_epoch = t + start    
        print(f"Epoch {real_epoch}\n-------------------------------")   
        train(train_dataloader, model, criterion, optimizer,real_epoch)
        test(test_dataloader, model)
    
    torch.save(model.state_dict(), MODEL_FILE)
    print("Done!")

if __name__ == "__main__":
    op_type = sys.argv[1]
    print(op_type)
    field = "f_high_mean_rate" #sys.argv[2] #"f_low_mean_rate" # next_low_rate, f_high_mean_rate, f_low_mean_rate
    if op_type == "training":
        # python seq_train_list.py training f_high_mean_rate
        training(field)
    if op_type == "tmp":
        scores = torch.tensor([[0.5, 2.0, 1.0], [0.9, -1.2, 0.0]])
        relevance = torch.tensor([[2, 0, 1], [0, 1, 0]])
        print(scores.shape)
        loss_fn = PairwiseHingeLoss() #ListMLELoss() #
        predict_scores = torch.tensor([[.1, .2, .3, 4, 70]])
        labels = torch.tensor([[10, 0, 0, 1, 5]])
        n = torch.tensor([5,1])
        loss = loss_fn(predict_scores, labels, n)
        
        loss = loss_fn(scores, relevance, torch.tensor([3,2]))
        
        print(loss)
        
    
    