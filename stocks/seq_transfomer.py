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
# print(f"Using {device} device")

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
            print(f"loss: {loss:>7f} , avg_loss: {avg_loss:>7f}  [{epoch:>5d}  {current:>5d}/{size:>5d}]") 
        
        if batch % 512 == 0:
            torch.save(model.state_dict(), MODEL_FILE+"."+str(epoch) + "." + str(int(batch / 512)) )
            # torch.save({
            # 'epoch': epoch,
            # 'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss.item(),
            # }, MODEL_FILE+"."+str(epoch))
            
    torch.save(model.state_dict(), MODEL_FILE+"."+str(epoch))
    torch.save(model.state_dict(), MODEL_FILE)
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
        for batch, (choose,reject) in enumerate(dataloader):         
            c = model(choose.to(device))
            r = model(reject.to(device))   
            loss = loss_fn(c, r)
            
            test_loss += loss.item()
            
    test_loss /= num_batches
    print(f"Test Avg loss: {test_loss:>8f} \n")

def training():
    # 初始化
    train_data = StockPairDataset("train","f_high_mean_rate")
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    # choose,reject = next(iter(train_dataloader))
    # print(choose.shape,reject.shape)

    test_data = StockPairDataset("validate","f_high_mean_rate")
    test_dataloader = DataLoader(test_data, batch_size=128)  
    
    criterion = LogExpLoss() #定义损失函数
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    
    learning_rate = 0.000005 #0.00001 #0.000005  #0.0000001  
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
        estimate_ndcg_score(dataloader=None,model=model)
        # scheduler.step()
    
    train_data.conn.close()
    test_data.conn.close()
    print("Done!")

# 完全随机：loss=0.703475

def predict():
    dataset = StockPredictDataset(predict_data_file="seq_predict.data")
    # print(next(iter(dataset)))
    dataloader = DataLoader(dataset, batch_size=128) 
     
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)

    if os.path.isfile(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE)) 
    
    model.eval()
    with torch.no_grad():
        all = []
        for _batch, (pk_date_stock,data) in enumerate(dataloader):         
            output = model(data.to(device))
            ret = list(zip(pk_date_stock.tolist(), output.tolist()))
            all = all + ret 
            # break 
        all.sort(key = lambda x:x[1],reverse=True)
        for item in all :
            print(";".join( [str(x) for x in item])) 

def compute_ndcg(df):
    ret = []
    date_groups = df.groupby(0)
    for date,data in date_groups:
        data = data.sort_values(by=[2])
        data[4] = [math.ceil((i+1)/3) for i in range(20)]
        
        data = data.sort_values(by=[3],ascending=False)
        mean_3 = data[2].head(3).mean()
        mean_all = data[2].mean() 
        
        y_true = np.expand_dims(data[4].to_numpy(),axis=0)
        y_predict = np.expand_dims(data[3].to_numpy(),axis=0)
        ndcg = ndcg_score(y_true,y_predict)
        ndcg_3 = ndcg_score(y_true,y_predict,k=3)
        
        ret.append([date,ndcg,ndcg_3,mean_3,mean_all])
    return ret     

def estimate_ndcg_score(dataloader=None,model=None):
    if dataloader is None:
        dataset = StockPointDataset()
        dataloader = DataLoader(dataset, batch_size=128) 
     
    if model is None:
        model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
        if os.path.isfile(MODEL_FILE):
            model.load_state_dict(torch.load(MODEL_FILE)) 
    
    model.eval()
    li_ = []
    with torch.no_grad():
        for _batch, (pk_date_stock,f_high_mean_rate,data) in enumerate(dataloader):         
            output = model(data.to(device))
            ret = list(zip(pk_date_stock.tolist(), f_high_mean_rate.tolist(), output.tolist())) 
            li_ = li_ + [(str(item[0])[:8], str(item[0])[8:], item[1], item[2]) for item in ret]
            # break
    
    # df = pd.read_csv("ndcg.txt",sep=";", header=None)
    df = pd.DataFrame(li_)
    ret = compute_ndcg(df) 
    
    print("ndcg:")
    print(sum([x[1] for x in ret])/len(ret))
    print(sum([x[2] for x in ret])/len(ret))
    print(sum([x[3] for x in ret])/len(ret))
    print(sum([x[4] for x in ret])/len(ret))

def evaluate_model_checkpoints():
    test_data = StockPairDataset("validate","f_high_mean_rate")
    test_dataloader = DataLoader(test_data, batch_size=128)   
    
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    criterion = LogExpLoss() #定义损失函数
    
    for i in range(31): #32
        fname =  MODEL_FILE + ".0."  + str(i) 
        print(fname)
        if os.path.isfile(fname):
            model.load_state_dict(torch.load(fname))
            test(test_dataloader, model, criterion) 

def gen_date_predict_scores():
    COMPARE_THRESHOLD = 0.02
    TOP_N = 10
    
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    if os.path.isfile(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))   
    model.eval()
    
     
    trade_dates = load_trade_dates(conn)
    for date in trade_dates:
        # print(date) 
        df = None
        data_file = "data/predict_results/%s.csv"%(date)
        if os.path.exists(data_file):
            df = pd.read_csv(data_file, sep=",", header=0, index_col=0)
            # print(df)
        else: 
            with torch.no_grad(): 
                dataset = StockPointDataset(datatype="train",trade_date=date)
                dataloader = DataLoader(dataset, batch_size=128) 
                # print(next(iter(dataloader)))
                
                all_ = []
                for _batch, (pk_date_stock,f_high_mean_rate,data) in enumerate(dataloader):         
                    output = model(data.to(device))
                    ret = list(zip(pk_date_stock.tolist(), f_high_mean_rate.tolist(), output.tolist()))
                    all_ = all_ + ret
                
                df = pd.DataFrame(all_,columns=['pk_date_stock','true_score','predict_score'])
                df = df.sort_values(by=["predict_score"],ascending=False)
                df.to_csv("data/predict_results/%s.csv"%(date))
        
        li = []
        li.append(date) 
        describe = df.describe()
        
        total_count = len(df)
        total_good = len(df.loc[df['true_score'] > COMPARE_THRESHOLD])
        li.append(total_count)
        li.append(total_good)
        li.append(round(total_good/total_count,2))
        
        top_10 = df.head(10)
        top_10_good =  len(top_10.loc[top_10['true_score'] > COMPARE_THRESHOLD])
        li.append(top_10_good)
        li.append(top_10_good/10)
        li.append(round(top_10['predict_score'].mean(),3))
        
        top_5 = df.head(5)
        top_5_good =  len(top_5.loc[top_5['true_score'] > COMPARE_THRESHOLD])
        li.append(top_5_good)
        li.append(top_5_good/5)
        li.append(round(top_5['predict_score'].mean(),3))
        
        top_n_5 = df.head(6).tail(5)
        top_n_5_good =  len(top_n_5.loc[top_n_5['true_score'] > COMPARE_THRESHOLD])
        li.append(top_n_5_good)
        li.append(top_n_5_good/5)
        li.append(round(top_n_5['predict_score'].mean(),3))
        
        li.append(round(describe["true_score"]["mean"],3))
        li.append(round(describe["true_score"]["std"],3))
        li.append(round(describe["predict_score"]["mean"],3))
        li.append(round(describe["predict_score"]["std"],3))

        print( ";".join([str(item) for item in li]))
        # break 
        

# python seq_transfomer.py training
# python seq_transfomer.py predict
if __name__ == "__main__": 
    # estimate_ndcg_score()
    # evaluate_model_checkpoints() 
    # gen_date_predict_scores()
    
    op_type = sys.argv[1]
    assert op_type in ("training", "predict")
    if op_type == "predict":
        predict()
    else:
        training()
