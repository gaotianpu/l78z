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

from seq_model_v3 import StockForecastModel,StockPointDataset,evaluate_ndcg_and_scores,SEQUENCE_LENGTH,D_MODEL,device

from pytorchltr.loss import PairwiseHingeLoss

# SEQUENCE_LENGTH = 20 #序列长度
# D_MODEL = 9  #维度
# MODEL_FILE = "model_list_stocks.pth"
MODEL_FILE = "model_list.pth" #默认dates

# https://blog.csdn.net/qq_36478718/article/details/122598406
# ListNet ・ ListMLE ・ RankCosine ・ LambdaRank ・ ApproxNDCG ・ WassRank ・ STListNet ・ LambdaLoss

def get_lr(train_steps, init_lr=0.1,warmup_steps=2500,max_steps=150000):
    """
    Implements gradual warmup, if train_steps < warmup_steps, the
    learning rate will be `train_steps/warmup_steps * init_lr`.
    Args:
        warmup_steps:warmup步长阈值,即train_steps<warmup_steps,使用预热学习率,否则使用预设值学习率
        train_steps:训练了的步长数
        init_lr:预设置学习率
    https://zhuanlan.zhihu.com/p/390261440
    
    """
    if warmup_steps and train_steps < warmup_steps:
        warmup_percent_done = train_steps / warmup_steps
        warmup_learning_rate = init_lr * warmup_percent_done  #gradual warmup_lr
        learning_rate = warmup_learning_rate
    else:
        # 这部分代码还有些问题
        #learning_rate = np.sin(learning_rate)  #预热学习率结束后,学习率呈sin衰减
        learning_rate = learning_rate**1.0001 #预热学习率结束后,学习率呈指数衰减(近似模拟指数衰减)
    return learning_rate 

class StockListDataset(Dataset):
    def __init__(self, datatype="train", field="f_high_mean_rate"):
        assert datatype in ("train", "validate", "test")
        # self.df = pd.read_csv("data2/list.stocks.%s_22223355.txt" % (datatype), sep=";", header=None)
        self.df = pd.read_csv("data3/list.date.%s_22223355.txt" % (datatype), sep=";", header=None)
        # self.df = pd.read_csv("data2/list.%s_22223355.txt" % (datatype), sep=";", header=None)
        self.conn = sqlite3.connect("file:data3/stocks_train_v3.db?mode=ro", uri=True)
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
        # predict_scores = model(torch.squeeze(data).to(device)) 
        # loss = loss_fn(torch.unsqueeze(predict_scores,0), labels) #8*8, 只关注头部1/8？, (1,8) 
        predict_scores = model(torch.flatten(data,end_dim=1).to(device)) 
        loss = loss_fn(predict_scores.reshape(labels.size()), labels)
        
        # Back propagation 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
        
        total_loss = total_loss + loss.item()
        
        if batch % 2048 == 0:
            avg_loss = total_loss / (batch + 1) 
            loss, current = loss.item(), (batch + 1) * len(labels)
            rate = round(current*100/size,2)
            print(f"loss: {loss:>7f} , avg_loss: {avg_loss:>7f}  [{epoch:>5d}  {current:>5d}/{size:>5d} {rate}%]")
                 
        cp_save_n = 2048 #cp, checkpoint
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
   
def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  
        
def training(field="f_high_mean_rate"):
    # 初始化
    train_data = StockListDataset(datatype="train",field=field)
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    
    test_data = StockPointDataset(datatype="test",field=field)
    test_dataloader = DataLoader(test_data, batch_size=128)  
    
    criterion = ListMLELoss() #
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    
    learning_r = 0.0000001 #0.00001 #0.000001  #0.0000001
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=learning_r, betas=(0.9,0.98), 
                                eps=1e-08) #定义最优化算法

    if os.path.isfile(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE)) 
        print("load success")

    model.to(device)
    
    epochs = 4
    start = 0
    for t in range(epochs):
        current_epoch = t + start + 1
        
        learning_r = 0.0000001
        if current_epoch==2:
            learning_r = 0.00001
        elif current_epoch in [3,4]:
            learning_r = 0.000001
        else:
            learning_r = 0.0000001
        adjust_learning_rate(optimizer, learning_r)
        
        print(f"Epoch={current_epoch}, lr={learning_r}\n-------------------------------")   
        train(train_dataloader, model, criterion, optimizer,current_epoch)
        # test(test_dataloader, model)
    
    torch.save(model.state_dict(), MODEL_FILE)
    print("Done!")

if __name__ == "__main__":
    op_type = sys.argv[1]
    print(op_type)
    field = "f_high_mean_rate" #sys.argv[2] #"f_low_mean_rate" # next_low_rate, f_high_mean_rate, f_low_mean_rate
    if op_type == "training":
        # python seq_train_list.py training
        training(field)
    if op_type == "test":
        # train_data = StockListDataset(datatype="train",field=field)
        # train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)
        
        # loss_fn = ListMLELoss() #
        # model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
        
        # for batch, (labels,data) in enumerate(train_dataloader):
        #     predict_scores = model(torch.flatten(data,end_dim=1).to(device)) 
        #     loss = loss_fn(predict_scores.reshape(labels.size()), labels)
        #     print(predict_scores.shape)
        #     print(predict_scores)
        #     print(predict_scores.reshape(labels.size()))
        #     print(loss)
            
        #     # print(labels.shape)
        #     # print(data.shape,data.size())
        #     # x = torch.flatten(data,end_dim=1)
        #     # print(x.shape)
        #     # x2 = x.reshape(data.size())
        #     # print(x2.shape)
        #     break 
    
        test_data = StockPointDataset(datatype="test",field=field)
        test_dataloader = DataLoader(test_data, batch_size=128)  
        
        model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
        mfile = "model_list_stocks.pth.3"
        if os.path.isfile(mfile):
            model.load_state_dict(torch.load(mfile)) 
            test(test_dataloader, model)
        
        mfile = "model_list_stocks.pth.4"
        if os.path.isfile(mfile):
            model.load_state_dict(torch.load(mfile)) 
            test(test_dataloader, model)
        
        for i in range(23):
            mfile = "model_list_stocks.pth.5." + str(i)
            if os.path.isfile(mfile):
                model.load_state_dict(torch.load(mfile)) 
                test(test_dataloader, model)
    
    if op_type == "tmp":
        scores = torch.tensor([[0.5, 2.0, 1.0], [0.9, -1.2, 0.0]])
        relevance = torch.tensor([[2, 0, 1], [0, 1, 0]])
        print(scores.shape)
        loss_fn = PairwiseHingeLoss() #ListMLELoss() #
        predict_scores = torch.tensor([[.1, .2, .3, 4, 70]])
        labels = torch.tensor([[10, 0, 0, 1, 5]])
        n = torch.tensor([5])
        # loss = loss_fn(predict_scores, labels, n)
        # print(loss)
        
        loss = loss_fn(scores, relevance, torch.tensor([3,2]))
        print(loss)
        
    
    