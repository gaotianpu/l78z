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

from model_v1 import BtcForecastModel,BtcPointDataset,SEQUENCE_LENGTH,D_MODEL,device

MODEL_TYPE = "high1" #high,low,high1,low1 
MODEL_FILE = f"model_point_{MODEL_TYPE}.pth" 

def get_model_output_file(model_type=MODEL_TYPE,data_type="test",epoch=1):
    return f"data/model_evaluate/point_{model_type}_{epoch}_{data_type}.txt"

def train(dataloader, model, loss_fn, optimizer,epoch): 
    size = len(dataloader.dataset) 
    
    model.train() #训练模式
    total_loss = 0.0 
    for batch, (pk_date_btc,true_scores,data) in enumerate(dataloader):         
        output = model(data.to(device))
        loss = loss_fn(output,true_scores)   
        
        # Back propagation 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  
        
        total_loss = total_loss + loss.item()
        
        if batch % 8 == 0:
            avg_loss = total_loss / (batch + 1) 
            loss, current = loss.item(), (batch + 1) * len(output)
            rate = round(current*100/size,2)
            print(f"loss: {loss:>8f} , avg_loss: {avg_loss:>8f}  [{epoch:>5d}  {current:>5d}/{size:>5d} {rate}%]") 
            
        # cp_save_n = 128 #cp, checkpoint
        # if batch % cp_save_n == 0:
        #     cp_idx = int(batch / cp_save_n)
        #     cp_idx_mod = cp_idx % 23
        #     torch.save(model.state_dict(), "%s.%s.%s" % (MODEL_FILE,epoch,cp_idx_mod) )
            
    torch.save(model.state_dict(), "%s.%s" % (MODEL_FILE,epoch))
    torch.save(model.state_dict(), MODEL_FILE)

def test(dataloader, model, loss_fn,data_type="test",epoch=0):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    
    model.eval()
    with torch.no_grad():
        all_ret = []
        for _batch, (pk_date_btc,true_scores,data) in enumerate(dataloader): 
            output = model(data.to(device))
            loss = loss_fn(output,true_scores) 
            test_loss += loss.item()
            
            # 准备计算分档loss，ndcg相关的数据
            ret = list(zip(pk_date_btc.tolist(), output.tolist(),true_scores.tolist()))
            all_ret = all_ret + ret    
    
    test_loss /= num_batches
    aft_test_loss = test_loss ** 0.5
    print(f"\n{data_type} Avg loss: {aft_test_loss:>8f} -> {test_loss:>8f}")
    
    df = pd.DataFrame(all_ret,columns=['pk_date_btc','predict_score','true_score'])
    df = df.sort_values(by=["predict_score"],ascending=False)
    df.to_csv(get_model_output_file(MODEL_TYPE,data_type,epoch),sep=";",index=False)

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  
        
def training(field="highN"):
    train_dataset = BtcPointDataset("train",field)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    validate_dataset = BtcPointDataset("validate",field)
    validate_dataloader = DataLoader(validate_dataset, batch_size=128) 
    
    criterion = nn.MSELoss() #均方差损失函数
    model = BtcForecastModel(SEQUENCE_LENGTH,D_MODEL) 
    learning_rate = 0.0000001 #0.0000001 #0.000001 #0.000001  #0.000001  
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=learning_rate, betas=(0.9,0.98), 
                                eps=1e-08) #定义最优化算法 

    if os.path.isfile(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))
        print("load success")
    model.to(device)

    epochs = 5
    start = 0
    for t in range(epochs):
        current_epoch = t+1+start 
        
        if current_epoch in [2,3]:
            learning_rate = 0.0001
        elif current_epoch == 4 :
            learning_rate = 0.000001
        else:
            learning_rate = 0.0000001
        adjust_learning_rate(optimizer, learning_rate)
        
        print(f"Epoch={current_epoch}, lr={learning_rate} \n-------------------------------")   
        train(train_dataloader, model, criterion, optimizer,current_epoch)
        test(validate_dataloader, model, criterion,"vaildate",current_epoch)
        
    
    torch.save(model.state_dict(), MODEL_FILE)
    print("Done!")
    
# python train_point.py training highN
if __name__ == "__main__":
    op_type = sys.argv[1]
    field = sys.argv[2] ## highN, lowN, low1, high1,
    print(op_type)
    if op_type == "training":
        MODEL_TYPE = field
        MODEL_FILE = f"model_point_{field}.pth" 
        training(field)
    
    

