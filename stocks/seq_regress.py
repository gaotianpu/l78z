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

from common import load_trade_dates
from seq_model import LogExpLoss,StockForecastModel,StockPairDataset,StockPointDataset,StockPredictDataset

SEQUENCE_LENGTH = 20 #序列长度
D_MODEL = 9  #维度
MODEL_FILE = "StockForecastModel.pth"

conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  #苹果的Metal Performance Shaders（MPS）
    if torch.backends.mps.is_available()
    else "cpu"
)

# 4. train 函数
def train(dataloader, model, loss_fn, optimizer,epoch): 
    size = len(dataloader.dataset) 
    
    model.train() #训练模式
    total_loss = 0.0 
    for batch, (pk_date_stock,true_scores,data) in enumerate(dataloader):         
        output = model(data.to(device))
        loss = loss_fn(output,true_scores)   
        
        # Back propagation 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  
        
        if batch % 64 == 0:
            avg_loss = total_loss / 64 #(batch + 1) 
            loss, current = loss.item(), (batch + 1) * len(output)
            print(f"loss: {loss:>7f} , avg_loss: {avg_loss:>7f}  [{epoch:>5d}  {current:>5d}/{size:>5d}]") 
            total_loss = loss.item()
        else:
            total_loss = total_loss + loss.item()
            
        
        if batch % 1024 == 0:
            torch.save(model.state_dict(), MODEL_FILE+"."+str(epoch) + "." + str(int(batch / 512)) )
            # torch.save({
            # 'epoch': epoch,
            # 'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss.item(),
            # }, MODEL_FILE+"."+str(epoch ))
            
    torch.save(model.state_dict(), MODEL_FILE+"."+str(epoch))
    
    # torch.save({
    #         'epoch': EPOCH,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss.item(),
    #         }, PATH+"."+str(epoch))

# 5. vaildate/test 函数
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    
    model.eval()
    with torch.no_grad():
        for _batch, (pk_date_stock,true_scores,data) in enumerate(dataloader): 
            output = model(data.to(device))
            loss = loss_fn(output,true_scores) 
            
            test_loss += loss.item()
            
    test_loss /= num_batches
    print(f"Test Avg loss: {test_loss:>8f} \n")

def training():
    # 初始化
    train_data = StockPointDataset(datatype="train")
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    # a = next(iter(train_dataloader))
    # print(choose.shape,reject.shape)

    test_data = StockPointDataset(datatype="validate")
    test_dataloader = DataLoader(test_data, batch_size=128)  
    
    criterion = nn.MSELoss() #均方差损失函数
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    
    learning_rate = 0.0000001 #0.00001 #0.000005  #0.0000001  
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

    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer,t)
        test(test_dataloader, model, criterion)
        # estimate_ndcg_score(dataloader=None,model=model)
        # scheduler.step()
        
    torch.save(model.state_dict(), MODEL_FILE)
    
    train_data.conn.close()
    test_data.conn.close()
    print("Done!")


 

# python seq_transfomer.py training
# python seq_transfomer.py predict
if __name__ == "__main__": 
    # estimate_ndcg_score()
    # evaluate_model_checkpoints() 
    # gen_date_predict_scores()
    training()
    
    # op_type = sys.argv[1]
    # assert op_type in ("training", "predict")
    # if op_type == "predict":
    #     predict()
    # else:
    #     training()
