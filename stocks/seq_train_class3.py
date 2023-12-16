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

from common import c_round,load_trade_dates
from seq_model_v4 import StockForecastModel,StockPointDataset,StockPredictDataset,evaluate_ndcg_and_scores,SEQUENCE_LENGTH,D_MODEL,device

MODEL_FILE = "model_cls3.pth"

conn = sqlite3.connect("file:data4/stocks_train_v4.db?mode=ro", uri=True)

class StockBinDataset(Dataset):
    def __init__(self, data_type="train", field="highN_rate"):
        assert data_type in ("train", "validate", "test")
        # dtmap = {"train":0,"validate":1,"test":2}
        # dataset_type = dtmap.get(data_type)
        # % (data_type,MODEL_TYPE)
        self.df = pd.read_csv("data4/three_train.txt", sep="|", header=None)
        self.conn = conn
        self.field = field  # 基于哪个预测值做比较

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = int(self.df.iloc[idx][0]) 
        pk_date_stock = self.df.iloc[idx][1] 
        
        sql = f"select pk_date_stock,data_json from stock_for_transfomer where pk_date_stock={pk_date_stock}"
        df_data = pd.read_sql(sql, self.conn)
        data_json = json.loads(df_data.iloc[0]["data_json"])
        past_days = torch.tensor(data_json["past_days"])
        
        return pk_date_stock,torch.tensor(label),past_days

# 4. train 函数
def train(dataloader, model, loss_fn, optimizer,epoch): 
    size = len(dataloader.dataset) 
    
    model.train() #训练模式
    total_loss = 0.0 
    for batch, (pk_date_stock,labels,data) in enumerate(dataloader):         
        output = model(data.to(device))
        loss = loss_fn(output, labels)
        
        # Back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
        
        total_loss = total_loss + loss.item()
        
        if batch % 64 == 0:
            avg_loss = total_loss / (batch + 1) 
            loss, current = loss.item(), (batch + 1) * len(labels)
            rate = round(current*100/size,2)
            print(f"loss: {loss:>7f} , avg_loss: {avg_loss:>7f}  [{epoch:>5d}  {current:>5d}/{size:>5d} {rate}%]") 
        
        cp_save_n = 128 #cp, checkpoint
        if batch % cp_save_n == 0:
            cp_idx = int(batch / cp_save_n)
            cp_idx_mod = cp_idx % 23
            torch.save(model.state_dict(), "%s.%s.%s" % (MODEL_FILE,epoch,cp_idx_mod) )
            
    torch.save(model.state_dict(), MODEL_FILE+"."+str(epoch)) 

# 5. vaildate/test 函数
def test(dataloader, model,data_type="test",epoch=0):
    df = None 
    fname = f"class3_{data_type}_{epoch}.txt"
    # rm -f class3_*.txt
    
    if os.path.exists(fname):
        print("####"+fname)
        df = pd.read_csv(fname,sep=";",header=0,dtype={'pk_date_stock':int})
    else:
        m = nn.Softmax(dim=1)
        model.eval()
        with torch.no_grad():
            all_ret = []
            for _batch, (pk_date_stock,true_scores,list_labels,data) in enumerate(dataloader): 
                output = m(model(data.to(device)))
                cls_idx = torch.argmax(output,dim=1)
                # print(cls_idx,output[:,1])
                
                # 准备计算分档loss，ndcg相关的数据
                ret = list(zip(pk_date_stock.tolist(),cls_idx.tolist(), output[:,1].tolist(),output[:,0].tolist(),output[:,2].tolist(),true_scores.tolist()))
                all_ret = all_ret + ret
                # break
            df = pd.DataFrame(all_ret,columns=["pk_date_stock","cls_idx","predict","predict_0","predict_2","true"])
            df = df.sort_values(by=["predict"],ascending=False)
            df.to_csv(fname,sep=";",index=False)
    
    # print(df)
            
    total = len(df)
    total_mean = c_round(df['true'].mean())
    df = df[df['cls_idx']==1]
    cls1_mean = c_round(df['true'].mean())
    true_threshold = 0.05
    print(f"{data_type} {epoch} total",c_round(len(df)/total),cls1_mean,total_mean)
    for predict_threshold in [0.55,0.58,0.6,0.65]: #0.38,0.4,0.45,0.5,
        r = df[df['predict']>predict_threshold]
        r1 = r[r['true']>true_threshold]
        tmp = c_round(len(r1)/len(r)) if len(r)>0 else "Nan"
        print(data_type,epoch,predict_threshold,c_round(len(r)/total),c_round(r['true'].mean()),tmp)    

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  
        
def training():
    # 初始化
    train_data = StockBinDataset("train","highN_rate")
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)

    # validate_data = StockPairDataset("validate","highN_rate")
    # validate_dataloader = DataLoader(validate_data, batch_size=128)  
    
    # test_data = StockPairDataset("test","highN_rate")
    # test_dataloader = DataLoader(test_data, batch_size=128)
    
    criterion = nn.CrossEntropyLoss() #定义损失函数
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL,3).to(device)
    
    learning_rate = 0.0000001 #0.0000001
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=learning_rate, betas=(0.9,0.98), 
                                eps=1e-08) #定义最优化算法
    
    if os.path.isfile(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))
        print("load success")

    model.to(device)
    
    epochs = 5
    start = 22
    for t in range(epochs):
        current_epoch = t+1+start
        if current_epoch in [2,3]:
            learning_rate = 0.0001
        elif current_epoch in [4,5]:
            learning_rate = 0.00001
        else:
            learning_rate = 0.00001
        adjust_learning_rate(optimizer, learning_rate)
        
        print(f"Epoch:{current_epoch}, lr={learning_rate}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer,current_epoch)
        # test(validate_dataloader, model, criterion, "validate")
        # test(test_dataloader, model, criterion, "test")
        # estimate_ndcg_score(ndcg_dataloader,model)
            # scheduler.step()
    
    torch.save(model.state_dict(), MODEL_FILE)
    print("Done!")

def evaluate_model_checkpoints(field="highN_rate"):
    '''用于检查哪个checkpoint效果更好些'''
    vali_data = StockPointDataset(datatype="validate",field=field)
    vali_dataloader = DataLoader(vali_data, batch_size=128)  
    
    test_data = StockPointDataset(datatype="test",field=field)
    test_dataloader = DataLoader(test_data, batch_size=128)
    
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL,3).to(device)
    
    for i in range(23,28):
        fname = f"{MODEL_FILE}.{i}"
        if not os.path.isfile(fname):
            print("\n### %s is not exist" % (fname))
            continue 
        
        print("\n###" + fname)
        model.load_state_dict(torch.load(fname))
        test(vali_dataloader, model,"vaildate",i)
        test(test_dataloader, model,"test",i)
        # break

def evaluate_real(trade_date):
    '''用于检查哪个checkpoint效果更好些''' 
    field="highN_rate"
    test_data = StockPredictDataset(f'data4/seq_predict/f_{trade_date}.data')
    test_dataloader = DataLoader(test_data, batch_size=128)
    
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL,3).to(device)
    
    for i in range(12,23):
        fname = f"{MODEL_FILE}.{i}"
        if not os.path.isfile(fname):
            print("\n### %s is not exist" % (fname))
            continue 
        
        print("\n###" + fname)
        model.load_state_dict(torch.load(fname)) 
        test(test_dataloader, model,f"real_{trade_date}",i)
        # break

# python seq_train_class3.py training
if __name__ == "__main__": 
    op_type = sys.argv[1]
    print("#",op_type)
    if op_type == "training": 
        training()
    if op_type == "evaluate_model_checkpoints":
        # python seq_train_class3.py evaluate_model_checkpoints
        evaluate_model_checkpoints()
    # if op_type == "gen_date_predict_scores_all":
    #     gen_date_predict_scores_all()

    if op_type == "evaluate_real":
        # python seq_train_class3.py evaluate_real 20231205
        date = sys.argv[2] #20231204
        evaluate_real(date)
