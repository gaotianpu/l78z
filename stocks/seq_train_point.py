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

from seq_model_v4 import StockForecastModel,StockPointDataset,evaluate_ndcg_and_scores,SEQUENCE_LENGTH,D_MODEL,device

MODEL_TYPE = "low1" #high,low,high1,low1 
MODEL_FILE = "model_point_%s.pth" % (MODEL_TYPE)

def get_model_output_file(model_type=MODEL_TYPE,data_type="test",epoch=1):
    return f"data4/model_evaluate/point_{model_type}_{epoch}_{data_type}.txt"

# 4. train 函数
def train(dataloader, model, loss_fn, optimizer,epoch): 
    size = len(dataloader.dataset) 
    
    model.train() #训练模式
    total_loss = 0.0 
    # pk_date_stock,true_scores,list_labels,data
    for batch, (pk_date_stock,true_scores,list_labels,data) in enumerate(dataloader):         
        output = model(data.to(device))
        loss = loss_fn(output,true_scores)   
        
        # Back propagation 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  
        
        total_loss = total_loss + loss.item()
        
        if batch % 128 == 0:
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
def test(dataloader, model, loss_fn,data_type="test",epoch=0):
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
    
    df = pd.DataFrame(all_ret,columns=['pk_date_stock','predict_score','true_score','list_label'])
    df = df.sort_values(by=["predict"],ascending=False)
    df.to_csv(get_model_output_file(MODEL_TYPE,data_type,epoch),sep=";",index=False)
    

def evaluate_label_loss(df):
    # 分档位loss trade_date = str(df_predict['pk_date_stock'][0])[:8]
    df["mse_loss"] = df.apply(lambda x: (x['predict'] - x['true'])**2 , axis=1)
    label_groups = df.groupby('label')
    for label,data in label_groups:
        mse_loss_mean = data['mse_loss'].mean()
        print(label,len(data), round(mse_loss_mean**0.5,4)) 
    # df.to_csv("data/test_label_loss.txt",sep=";",index=False)

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr        

def training(field="highN_rate"):
    # 初始化
    train_data = StockPointDataset(datatype="train",field=field)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

    # vali_data = StockPointDataset(datatype="validate",field=field)
    # vali_dataloader = DataLoader(vali_data, batch_size=128)  
    
    # test_data = StockPointDataset(datatype="test",field=field)
    # test_dataloader = DataLoader(test_data, batch_size=128)  
    
    criterion = nn.MSELoss() #均方差损失函数
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL)
    
    learning_rate = 0.0000001 #0.0000001 #0.000001 #0.000001  #0.000001  
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
        
    model.to(device)

    epochs = 4
    start = 0
    for t in range(epochs):
        current_epoch = t+1+start 
        
        if current_epoch in [2,3]:
            learning_rate = 0.0001
        elif current_epoch in [4]:
            learning_rate = 0.00001
        else:
            learning_rate = 0.00001
        adjust_learning_rate(optimizer, learning_rate)
        
        print(f"Epoch={current_epoch}, lr={learning_rate} \n-------------------------------")   
        train(train_dataloader, model, criterion, optimizer,current_epoch)
        # if current_epoch>3:
        #     test(vali_dataloader, model, criterion,"vaildate",current_epoch)
        #     test(test_dataloader, model, criterion,"test",current_epoch)
        # scheduler.step()
    
    torch.save(model.state_dict(), MODEL_FILE)
    print("Done!")

def evaluate_low1(dataset_type="test"):
    '''评估《最低价预测模型》 的效果
    1. 按照预测的最低价买入，成功率多少； 
    2. 成功部分，与实际的最低价比，平均差值多少
    3. 指定条件下(例如，指定其他模型阈值下)，上述二者效果时多少
    '''
    t_data = StockPointDataset(dataset_type,field="next_low_rate")
    t_dataloader = DataLoader(t_data, batch_size=128)
    
    for i in range(0,6):
        df = None
        output_file = output_file = get_model_output_file("low1",dataset_type,i)
        if not os.path.exists(output_file):
            print(output_file, "is not exist")
        
        df = pd.read_csv(output_file,sep=";",header=0)
        df_buy_success = df[df['predict_score']>df['true_score']]
        df_buy_success['diff'] = abs(df['predict_score']-df['true_score'])
        print(dataset_type,i,len(df),len(df_buy_success),len(df_buy_success)/len(df),df_buy_success['diff'].mean())
        

def evaluate_model_checkpoints(field="highN_rate"):
    '''用于检查哪个checkpoint效果更好些'''
    vali_data = StockPointDataset(datatype="validate",field=field)
    vali_dataloader = DataLoader(vali_data, batch_size=128)  
    
    test_data = StockPointDataset(datatype="test",field=field)
    test_dataloader = DataLoader(test_data, batch_size=128)
    
    # high,low,high1,low1 
    maps = {'highN_rate':'high','next_high_rate':'high1','next_low_rate':'low1'}
    
    criterion = nn.MSELoss() #均方差损失函数
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    for i in range(4,6):
        model_name = maps.get(field)
        fname = f"model_point_{model_name}.pth.{i}"
        if not os.path.isfile(fname):
            print("\n### %s is not exist" % (fname))
            continue 
        
        print("\n###" + fname)
        model.load_state_dict(torch.load(fname))
        test(vali_dataloader, model, criterion,"vaildate")
        test(test_dataloader, model, criterion,"test")
        break
                    

# python seq_train_point.py training highN_rate
if __name__ == "__main__":
    op_type = sys.argv[1]
    field = sys.argv[2] ## highN_rate, next_low_rate, next_high_rate
    print(op_type)
    if op_type == "training":
        training(field)
    if op_type == "checkpoints": 
        # python seq_train_point.py checkpoints highN_rate
        evaluate_model_checkpoints(field)