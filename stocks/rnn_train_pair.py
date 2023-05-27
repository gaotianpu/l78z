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
class StockPairDataset(Dataset):
    def __init__(self,data_type="train"):
        assert data_type in ('train','validate','predict')
        self.pd_frame = pd.read_csv("data/rnn_%s.txt"%(data_type),sep=";",header=None)
        self.df_pairs = pd.read_csv("data/rnn_%s_pair.txt"%(data_type),
            sep=" ",names="label,index_0,index_1".split(","),header=None)

    def __len__(self):
        return len(self.df_pairs)

    def __getitem__(self, idx):
        pair_info = self.df_pairs.iloc[idx]
        label,idx_0,idx_1 = pair_info['label'],pair_info['index_0'],pair_info['index_1']
        label = np.float32(np.array([label]))

        X_0 = self.pd_frame.iloc[idx_0, 6:]
        X_1 = self.pd_frame.iloc[idx_1, 6:]

        X_0 = [[float(x) for x in  v.split(",")] for v in X_0.tolist()]
        X_1 = [[float(x) for x in  v.split(",")] for v in X_1.tolist()]

        # # [::-1], 时间序列倒序
        X_0 = np.float32(np.array(X_0)[::-1]).copy()
        X_1 = np.float32(np.array(X_1)[::-1]).copy()

        return X_0,X_1,label


# 2. 定义模型
class PairWiseModel(nn.Module):
    def __init__(self, input_dim, sequence_len, hidden_size, output_dim):
        super(PairWiseModel, self).__init__()
        num_layers = 2
        self.rnn = nn.GRU(input_dim, hidden_size, num_layers)

        self.fc = nn.Sequential(
             nn.Dropout(), #dropout,AlexNet引入
             nn.Linear(hidden_size, 10),
             nn.ReLU(),
             nn.Dropout(),
             nn.Linear(10, output_dim)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x0,x1):
        out0, _ = self.rnn(x0) 
        out0 = self.fc(out0[:, -1, :]) 

        out1, _ = self.rnn(x1) 
        out1 = self.fc(out1[:, -1, :])

        prob = self.sigmoid(out0-out1)
        return prob 
    
    def predict(self,x0):
        out0, _ = self.rnn(x0)
        out0 = self.fc(out0[:, -1, :]) 
        return out0

# 2.1 模型测试
model = PairWiseModel(9, 10, 30,1)

# 3. 设置训练参数
learning_rate = 0.05
criterion = nn.BCELoss() #定义损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #定义最优化算法

# 4. 训练
def train(dataloader, model, loss_fn, optimizer,epoch):
    # model = torch.load('model/rnn_model_0.pth')
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    model.train() #训练模式

    train_loss, train_correct = 0.0, 0
    for batch, (X_0,X_1,label) in enumerate(dataloader):         
        pred = model(X_0,X_1)
        loss = loss_fn(pred,label) 

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        label_preds = torch.round(pred)
        correct = torch.eq(label_preds, label).float()
        acc = correct.sum() / len(correct)

        # print("pred:", pred)
        # print("label:", label)
        # print("correct:", acc)
        train_loss += loss.item() 
        train_correct += acc # (pred>0.5 == label).type(torch.float).sum().item()

        if batch % 64 == 0:
            loss, current = loss.item(), batch * len(label)
            print(f"{epoch}-{batch}, loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # if batch>30:
        # break 
    train_loss = train_loss/num_batches   
    train_correct = train_correct/num_batches   
    print(f"train: {epoch}-{batch}, Avg loss: {train_loss:>7f} , Avg correct:{train_correct:>7f}")


# 5. 测试
def validate(dataloader, model, loss_fn,epoch):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() #预测模式
    test_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for X_0,X_1,label in dataloader:
#             X, y = X.to(device), y.to(device)
            pred = model(X_0,X_1)
            # print("pred:",pred, pred.type)
            # print("label:",label,label.type)
            # print("loss:",loss_fn(pred, label).item())
            label_preds = torch.round(pred)
            correct = torch.eq(label_preds, label).float()
            acc = correct.sum() / len(correct)

            test_loss += loss_fn(pred, label).item() 
            val_correct += acc
    test_loss /= num_batches
    val_correct /= num_batches
    print(f"test, {epoch}, Avg loss: {test_loss:>8f} correct:{val_correct} \n")


# 6. 启动训练
def start_train():   
    trainData = StockPairDataset("train")
    train_dataloader = DataLoader(trainData, batch_size=64, shuffle=True)

    validateData = StockPairDataset("validate")
    validate_dataloader = DataLoader(validateData, batch_size=64, shuffle=False)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer,t)
        validate(validate_dataloader, model, criterion,t)
        torch.save(model, 'model_tmp/rnn_pair_%d.pth'%(t)) 
         
    print("Done!")

def test():
    validate = StockPairDataset("validate")
    test_dataloader = DataLoader(validate, batch_size=32, shuffle=False)

    # 测试
    print("len(validate):",len(validate))
    print("validate[0]:", validate[0])

    # train_features, train_y = next(iter(test_dataloader))
    # print("train_features:",train_features,train_features.size())
    # print("train_y:",train_y,train_y.size())

if __name__ == "__main__":
    # test()
    start_train()