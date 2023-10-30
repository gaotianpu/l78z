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

from seq_model_v2 import StockForecastModel,StockPointDataset,evaluate_ndcg_and_scores,SEQUENCE_LENGTH,D_MODEL,device

MODEL_TYPE = "low" #high,low,high1,low1
MODEL_FILE = "model_point_%s.pth" % (MODEL_TYPE)

# 4. train 函数
def train(dataloader, model, loss_fn, optimizer,epoch): 
    size = len(dataloader.dataset) 
    
    model.train() #训练模式
    total_loss = 0.0 
    for batch, (pk_date_stock,true_scores,list_label,data) in enumerate(dataloader):         
        output = model(data.to(device))
        loss = loss_fn(output,true_scores)   
        
        # Back propagation 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  
        
        total_loss = total_loss + loss.item()
        
        if batch % 64 == 0:
            avg_loss = total_loss / (batch + 1) 
            loss, current = loss.item(), (batch + 1) * len(output)
            rate = round(current*100/size,2)
            print(f"loss: {loss:>7f} , avg_loss: {avg_loss:>7f}  [{epoch:>5d}  {current:>5d}/{size:>5d} {rate}%]") 
            
        cp_save_n = 1280 #cp, checkpoint
        if batch % cp_save_n == 0:
            cp_idx = int(batch / cp_save_n)
            cp_idx_mod = cp_idx % 23
            torch.save(model.state_dict(), "%s.%s.%s" % (MODEL_FILE,epoch,cp_idx_mod) )
            
    torch.save(model.state_dict(), "%s.%s" % (MODEL_FILE,epoch))
    
    # torch.save({
    #         'epoch': EPOCH,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss.item(),
    #         }, PATH+"."+str(epoch))

# 5. vaildate/test 函数
def test(dataloader, model, loss_fn,data_type="test"):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    
    model.eval()
    with torch.no_grad():
        all_ret = []
        for _batch, (pk_date_stock,true_scores,list_label,data) in enumerate(dataloader): 
            output = model(data.to(device))
            loss = loss_fn(output,true_scores) 
            test_loss += loss.item()
            
            # 准备计算分档loss，ndcg相关的数据
            ret = list(zip(pk_date_stock.tolist(), output.tolist(),true_scores.tolist(),list_label.tolist()))
            all_ret = all_ret + ret      
    
    test_loss /= num_batches
    test_loss = test_loss ** 0.5
    print(f"\n{data_type} Avg loss: {test_loss:>8f}")
    
    df = pd.DataFrame(all_ret,columns=["pk_date_stock","predict","true","label"])
    # df["trade_date"] = df.apply(lambda x: str(x['pk_date_stock'])[:8] , axis=1)
    df = evaluate_ndcg_and_scores(df)
    evaluate_label_loss(df)
    

def evaluate_label_loss(df):
    # 分档位loss trade_date = str(df_predict['pk_date_stock'][0])[:8]
    df["mse_loss"] = df.apply(lambda x: (x['predict'] - x['true'])**2 , axis=1)
    label_groups = df.groupby('label')
    for label,data in label_groups:
        mse_loss_mean = data['mse_loss'].mean()
        print(label,len(data), round(mse_loss_mean**0.5,4)) 
    # df.to_csv("data/test_label_loss.txt",sep=";",index=False)
        

def training(field="f_high_mean_rate"):
    # 初始化
    train_data = StockPointDataset(datatype="train",field=field)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

    vali_data = StockPointDataset(datatype="validate",field=field)
    vali_dataloader = DataLoader(vali_data, batch_size=128)  
    
    test_data = StockPointDataset(datatype="test",field=field)
    test_dataloader = DataLoader(test_data, batch_size=128)  
    
    criterion = nn.MSELoss() #均方差损失函数
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    
    learning_rate = 0.00001 #0.00001 #0.000001  #0.0000001  
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=learning_rate, betas=(0.9,0.98), 
                                eps=1e-08) #定义最优化算法
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    if os.path.isfile(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE)) 
        # checkpoint = torch.load(MODEL_FILE)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        print("load success")

    epochs = 2
    start = 1
    for t in range(epochs):
        current_epochs = t+1+start 
        print(f"Epoch {current_epochs}\n-------------------------------")   
        train(train_dataloader, model, criterion, optimizer,current_epochs)
        test(vali_dataloader, model, criterion,"vaildate")
        test(test_dataloader, model, criterion,"test")
        # scheduler.step()
    
    torch.save(model.state_dict(), MODEL_FILE)
    print("Done!")


def evaluate_model_checkpoints(field="f_high_mean_rate"):
    '''用于检查哪个checkpoint效果更好些'''
    vali_data = StockPointDataset(datatype="validate",field=field)
    vali_dataloader = DataLoader(vali_data, batch_size=128)  
    
    test_data = StockPointDataset(datatype="test",field=field)
    test_dataloader = DataLoader(test_data, batch_size=128)
    
    criterion = nn.MSELoss() #均方差损失函数
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    
    for i in range(23): #32
        fname =  "%s.1.%s" % (MODEL_FILE,17)
        fname = "model_point_low.pth.3.4"
        if not os.path.isfile(fname):
            print("\n### %s is not exist" % (fname))
            continue 
        
        print("\n###" + fname)
        model.load_state_dict(torch.load(fname))
        test(vali_dataloader, model, criterion,"vaildate")
        test(test_dataloader, model, criterion,"test")
        break
                    

# python seq_train_point.py training f_high_mean_rate
if __name__ == "__main__":
    op_type = sys.argv[1]
    field = sys.argv[2] ## f_high_mean_rate,f_low_mean_rate, next_low_rate, next_high_rate
    print(op_type)
    if op_type == "training":
        training(field)
    if op_type == "checkpoints": 
        # python seq_train_point.py checkpoints f_high_mean_rate
        evaluate_model_checkpoints(field)