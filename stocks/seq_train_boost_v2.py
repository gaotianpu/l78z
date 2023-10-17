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

from seq_model_v2 import evaluate_ndcg_and_scores

MODEL_FILE = "model_boost_point.pth" # model_boost_point,model_boost_point_high1,model_boost_point_low1

D_MODEL = 8  #带true_open_rate

# conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  #苹果的Metal Performance Shaders（MPS）
    if torch.backends.mps.is_available()
    else "cpu"
)

class StockPointDataset(Dataset):
    def __init__(self,datatype="validate",trade_date=None, y_field="true_score"): 
        dtmap = {"train":0,"validate":1,"test":2}
        assert y_field in "true_score,true_high_1,true_low_1".split(",")
        self.y_field = y_field
        self.conn = sqlite3.connect("file:data/stocks_train_4.db?mode=ro", uri=True)
        dataset_type = dtmap.get(datatype) 
        self.min_max = {'pair_15_min': -2.4835965633392334, 'pair_15_max': 4.631412982940674, 'list_235_min': -0.2671133875846863, 'list_235_max': 2.955233097076416, 'point_5_min': 0.05888749286532402, 'point_5_max': 0.32927578687667847, 'point_4_min': 0.0010410360991954803, 'point_4_max': 0.2968159019947052, 'pair_11_min': -1.5444180965423584, 'pair_11_max': 2.854917287826538, 'point_high1_min': -0.48987817764282227, 'point_high1_max': 0.25763288140296936, 'low1_min': -0.13178201019763947, 'low1_max': -0.005453449673950672, 'true_open_rate_min': -0.7574999928474426, 'true_open_rate_max': 3.8977999687194824}
        self.x_fields = "pair_15,list_235,point_5,point_4,pair_11,point_high1,low1,true_open_rate".split(",")
        self.df = pd.read_csv('data/seq_train_%s.txt' % (dataset_type),header=None)
    
    def normal(self,item,field_name):
        min_val = self.min_max.get(field_name+"_min")
        max_val = self.min_max.get(field_name+"_max")
        return (item[field_name] - min_val)/(max_val-min_val)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pk_date_stock = self.df.iloc[idx][0] 
        sql = "select * from stock_for_boost_v2 where pk_date_stock=%s" % (pk_date_stock)
        df_item = pd.read_sql(sql, self.conn) 
        
        item = df_item.iloc[0]
        list_label = item['list_label'] 
        
        # y_field: true_score,true_high_1,true_low_1
        y_value = torch.tensor(item[self.y_field],dtype=torch.float32)
        
        # true_open_rate: 用作当日开盘拿到开盘价后的预测依据
        # [item['pair_15'],
        # item['list_235'],
        # item['point_5'],
        # item['point_4'],
        # item['pair_11'],
        # item['point_high1'],
        # item['low1'],
        # item['true_open_rate']],
        
        
        fields = torch.tensor([self.normal(item,field) for field in self.x_fields],
                              dtype=torch.float32)
        # pk_date_stock,true_scores,list_label,data #兼容计算ndcg的逻辑
        return pk_date_stock, y_value, list_label, fields

class StockBoostModel(nn.Module):
    def __init__(self, d_model: int = 8) -> None:
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        scores = self.linear_relu_stack(x)
        return scores.squeeze(1)  #

# 4. train 函数
def train(dataloader, model, loss_fn, optimizer,epoch): 
    size = len(dataloader.dataset) 
    
    model.train() #训练模式
    total_loss = 0.0 
    for batch, (pk_date_stock,true_scores,list_label,data) in enumerate(dataloader):         
        output = model(data.to(device))
        # print(output)
        # print(output.shape)
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
    print(f"{data_type} Avg loss: {test_loss:>8f}")
    
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
        

def training(field="true_score"):
    # 初始化
    train_data = StockPointDataset(datatype="train",y_field=field)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

    vali_data = StockPointDataset(datatype="validate",y_field=field)
    vali_dataloader = DataLoader(vali_data, batch_size=128)  
    
    test_data = StockPointDataset(datatype="test",y_field=field)
    test_dataloader = DataLoader(test_data, batch_size=128)  
    
    criterion = nn.MSELoss() #均方差损失函数
    model = StockBoostModel(D_MODEL).to(device)
    
    learning_rate = 0.0000001 #0.00001 #0.000001  #0.0000001  
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
    start = 0
    for t in range(epochs):
        current_epochs = t+1+start 
        print(f"Epoch {current_epochs}\n-------------------------------")   
        train(train_dataloader, model, criterion, optimizer,current_epochs)
        test(vali_dataloader, model, criterion,"vaildate")
        test(test_dataloader, model, criterion,"test")
        # scheduler.step()
    
    torch.save(model.state_dict(), MODEL_FILE)
    print("Done!")


def evaluate_model_checkpoints(field="true_score"):
    '''用于检查哪个checkpoint效果更好些'''
    vali_data = StockPointDataset(datatype="validate",y_field=field)
    vali_dataloader = DataLoader(vali_data, batch_size=128)  
    
    test_data = StockPointDataset(datatype="test",y_field=field)
    test_dataloader = DataLoader(test_data, batch_size=128)
    
    criterion = nn.MSELoss() #均方差损失函数
    model = StockBoostModel(D_MODEL).to(device)
    
    for i in range(4): #32
        fname =  "%s.1.%s" % (MODEL_FILE,i)
        if not os.path.isfile(fname):
            print("\n### %s is not exist" % (fname))
            continue 
        
        print("\n### " + fname)
        model.load_state_dict(torch.load(fname))
        test(vali_dataloader, model, criterion,"vaildate")
        test(test_dataloader, model, criterion,"test")
        
                    

# python seq_train_boost_v2.py training true_score
if __name__ == "__main__":
    op_type = sys.argv[1]
    field = sys.argv[2] ## true_score,true_high_1,true_low_1
    print(op_type)
    if op_type == "training":
        training(field)
        
        # train_data = StockPointDataset(datatype="train",y_field=field)
        # train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
        
        # criterion = nn.MSELoss() #均方差损失函数
        # model = StockBoostModel(D_MODEL).to(device)
        
        # learning_rate = 0.0000001 #0.00001 #0.000001  #0.0000001  
        # optimizer = torch.optim.Adam(model.parameters(), 
        #                             lr=learning_rate, betas=(0.9,0.98), 
        #                             eps=1e-08) #定义最优化算法
        # train(train_dataloader, model, criterion, optimizer,1)
    
    if op_type == "checkpoints": 
        # python seq_train_point.py checkpoints true_score
        evaluate_model_checkpoints(field)
        
        
        
# loss: 0.002989 , avg_loss: 0.002309  [    1  6881344/6884800 99.95%]
# vaildate Avg loss: 0.056446
# ndcg_scores:n=0.8462,n5=0.5936,n3=0.5684 , true_rate:t=0.0432,t5=0.0462,t3=0.0454
# 1 5805 0.0632
# 2 5805 0.0176
# 3 5811 0.0063
# 4 5814 0.0126
# 5 5814 0.0238
# 6 5814 0.0359
# 7 5814 0.0566
# 8 5811 0.1252
# test Avg loss: 0.047308
# ndcg_scores:n=0.7997,n5=0.5155,n3=0.4778 , true_rate:t=0.0205,t5=0.0223,t3=0.0219
# 1 11422 0.0579
# 2 11832 0.0176
# 3 5887 0.0063
# 4 5800 0.0126
# 5 2820 0.0237
# 6 2965 0.036
# 7 2893 0.0566
# 8 2893 0.1249