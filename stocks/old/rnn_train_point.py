#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from tkinter.ttk import Separator
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 1. 定义数据集
class StockDataset(Dataset):
    def __init__(self,data_type="sample_train"):
        assert data_type in ('sample_train','train','validate','predict')
        self.pd_frame = pd.read_csv("data/rnn_%s.txt"%(data_type),sep=";",header=None)

    def __len__(self):
        return len(self.pd_frame)

    def __getitem__(self, idx):
        trade_date = self.pd_frame.iloc[idx, 0]
        stock_no = self.pd_frame.iloc[idx, 1]
        y_high = self.pd_frame.iloc[idx, 2]
        y_low = self.pd_frame.iloc[idx, 3]
        label_high = self.pd_frame.iloc[idx, 4]
        label_low = self.pd_frame.iloc[idx, 5]

        Y = np.float32(np.array([y_high]))

        X = self.pd_frame.iloc[idx, 6:]
        X = [[float(x) for x in  v.split(",")] for v in X.tolist()]
        # [::-1], 时间序列倒序
        X = np.float32(np.array(X)[::-1]).copy()
        return X,Y,stock_no,trade_date

# 1.1 数据集
# 定义


# 测试
# print("len(train):",len(train))
# print("train[0]:", train[0])
# train_features, train_y = next(iter(train_dataloader))
# print("train_features:",train_features,train_features.size())
# print("train_y:",train_y,train_y.size())


# 2. 定义模型
class PointWiseModel(nn.Module):
    def __init__(self, input_dim, sequence_len, hidden_size, output_dim):
        super(PointWiseModel, self).__init__()
        num_layers = 2
        self.rnn = nn.GRU(input_dim, hidden_size, num_layers)
        # self.linear = nn.Linear(hidden_size, output_dim, bias=True)  

        self.fc = nn.Sequential(
             nn.Dropout(), #dropout,AlexNet引入
             nn.Linear(hidden_size, 10),
             nn.ReLU(),
             nn.Dropout(),
             nn.Linear(10, output_dim)
         )

    def forward(self, x):
        out, _ = self.rnn(x) 
        # out[:, -1, :], 选取最后一个时间点的 out 输出
        out = self.fc(out[:, -1, :]) 
        # print("out1:",out,out.size())
        return out

# 2.1 模型测试
model = PointWiseModel(9, 10, 30,1) 
# y = model(train_features)
# print("y:",y)

# 3. 设置训练参数
learning_rate = 0.05
criterion = nn.MSELoss() #定义损失函数：均方误差
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #定义最优化算法

# 4. 训练
def train(dataloader, model, loss_fn, optimizer,epoch):
    # model = torch.load('model/rnn_model_0.pth')

    size = len(dataloader.dataset)
    model.train() #训练模式
    for batch, (X, y,stock_no,trade_date) in enumerate(dataloader):         
        pred = model(X)
        loss = loss_fn(pred,y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"{epoch}-{batch}, loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
    print(f"train: {epoch}-{batch}, loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# 4.1 训练测试
# train(train_dataloader, model, criterion, optimizer)

# 5. 测试
def test(dataloader, model, loss_fn,epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() #预测模式
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y, stock_no,trade_date in dataloader:
#             X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    # correct /= size
    print(f"test, {epoch}, Avg loss: {test_loss:>8f} \n")


# 6. 启动训练
def start_train():   
    train_data = StockDataset("sample_train")
    test_data = StockDataset("validate")
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer,t)
        test(test_dataloader, model, criterion,t)
        torch.save(model, 'model_tmp/rnn_point_%d.pth'%(t))
    print("Done!")

if __name__ == "__main__":
    start_train()