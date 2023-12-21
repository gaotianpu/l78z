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

MODEL_TYPE = "date" # date,stock,date_stock
MODEL_FILE = "model_pair.pth"  #% (MODEL_TYPE)

class BtcPairDataset(Dataset):
    def __init__(self,datatype="train",field="highN_rate"):
        self.field = field
        dtmap = {"train":0,"validate":1,"test":2,"predict":3}
        dataset_type = dtmap.get(datatype)
        self.conn = sqlite3.connect("file:data/btc_train.db?mode=ro", uri=True)
        
        self.df_pairs = pd.read_csv(f"data/pairs_{dataset_type}_{field}.csv",sep=";",header=0)
        
        
    def __len__(self):
        return len(self.df_pairs)

    def __getitem__(self, idx):
        id1 = self.df_pairs.iloc[idx]['id_1']
        id2 = self.df_pairs.iloc[idx]['id_2']
        
        sql = f"select pk_date_btc,data_json from train_data where pk_date_btc in ({id1},{id2})"
        df_pair = pd.read_sql(sql, self.conn)
        a = json.loads(df_pair.iloc[0]["data_json"])
        b = json.loads(df_pair.iloc[1]["data_json"])
        a_t = torch.tensor(a["past_days"])
        b_t = torch.tensor(b["past_days"]) 
        if a[self.field] > b[self.field]:
            return a_t, b_t
        else:
            return b_t, a_t

class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """
    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss
    
# 4. train 函数
def train(dataloader, model, loss_fn, optimizer,epoch): 
    size = len(dataloader.dataset) 
    
    model.train() #训练模式
    total_loss = 0.0 
    for batch, (choose,reject) in enumerate(dataloader):         
        c = model(choose.to(device))
        r = model(reject.to(device))  
        
        loss = loss_fn(c, r)   
        
        # Back propagation 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
        
        total_loss = total_loss + loss.item()
        
        if batch % 64 == 0:
            avg_loss = total_loss / (batch + 1) 
            loss, current = loss.item(), (batch + 1) * len(choose)
            rate = round(current*100/size,2)
            print(f"loss: {loss:>7f} , avg_loss: {avg_loss:>7f}  [{epoch:>5d}  {current:>5d}/{size:>5d} {rate}%]") 
        
        cp_save_n = 1280 #cp, checkpoint
        if batch % cp_save_n == 0:
            cp_idx = int(batch / cp_save_n)
            cp_idx_mod = cp_idx % 23
            torch.save(model.state_dict(), "%s.%s.%s" % (MODEL_FILE,epoch,cp_idx_mod) )
            
    torch.save(model.state_dict(), MODEL_FILE+"."+str(epoch)) 
    torch.save(model.state_dict(), MODEL_FILE)

# 5. vaildate/test 函数
def test(dataloader, model, loss_fn, data_type="test",epoch=0):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    
    model.eval()
    with torch.no_grad():
        for batch, (choose,reject) in enumerate(dataloader):         
            c = model(choose.to(device))
            r = model(reject.to(device))   
            loss = loss_fn(c, r)
            
            test_loss += loss.item()
            
    test_loss /= num_batches
    print(f"{data_type} Avg loss: {test_loss:>8f}")

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  
        
def training():
    # 初始化
    train_data = BtcPairDataset("train","highN_rate")
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True) 

    validate_data = BtcPairDataset("validate","highN_rate")
    validate_dataloader = DataLoader(validate_data, batch_size=128)  
    
    # test_data = StockPairDataset("test","highN_rate")
    # test_dataloader = DataLoader(test_data, batch_size=128)
    
    criterion = LogExpLoss() #定义损失函数
    model = BtcForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    
    learning_rate = 0.00001 #0.0001 #0.00001 #0.000001
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
        print(f"Epoch={current_epoch}, lr={learning_rate}\n-------------------------------")
        
        if current_epoch in [2,3]:
            learning_rate = 0.0001
        elif current_epoch == 4 :
            learning_rate = 0.000001
        else:
            learning_rate = 0.0000001
        adjust_learning_rate(optimizer, learning_rate)
        
        train(train_dataloader, model, criterion, optimizer,current_epoch)
        test(validate_dataloader, model, criterion, "validate",current_epoch) 
        
    print("Done!")
        
def test_tmp():
    dataset = BtcPairDataset("test",field="highN_rate")
    a_t, b_t = next(iter(dataset))
    print(a_t)
    print(b_t)
    
if __name__ == "__main__":
    op_type = sys.argv[1]
    print(op_type)
    if op_type == "training":
        training()