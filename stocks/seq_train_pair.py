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
from seq_model_v4 import StockForecastModel,StockPointDataset,evaluate_ndcg_and_scores,SEQUENCE_LENGTH,D_MODEL,device

MODEL_TYPE = "date" # date,stock,date_stock
MODEL_FILE = "model_pair.pth"  #% (MODEL_TYPE)

conn = sqlite3.connect("file:data4/stocks_train_v4.db?mode=ro", uri=True)

class StockPairDataset(Dataset):
    def __init__(self, data_type="train", field="highN_rate"):
        assert data_type in ("train", "validate", "test")
        # dtmap = {"train":0,"validate":1,"test":2}
        # dataset_type = dtmap.get(data_type)
        self.df = pd.read_csv("data4/pair.%s.%s.txt" % (data_type,MODEL_TYPE), sep=";", header=None)
        self.conn = conn #sqlite3.connect("file:data4/stocks_train_v4.db?mode=ro", uri=True)
        self.field = field  # 基于哪个预测值做比较

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sql = (
            "select pk_date_stock,data_json from stock_for_transfomer where pk_date_stock in (%s,%s)"
            % (self.df.iloc[idx][0], self.df.iloc[idx][1])
        )
        df_pair = pd.read_sql(sql, self.conn)
        a = json.loads(df_pair.iloc[0]["data_json"])
        b = json.loads(df_pair.iloc[1]["data_json"])
        a_t = torch.tensor(a["past_days"])
        b_t = torch.tensor(b["past_days"]) 
        if a[self.field] > b[self.field]:
            return a_t, b_t
        else:
            return b_t, a_t

# 2. pair形式的损失函数
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
            
        # if batch % 512 == 0:
        #     torch.save(model.state_dict(), MODEL_FILE+"."+str(epoch) + "." + str(int(batch / 512)) )
            # torch.save({
            # 'epoch': epoch,
            # 'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss.item(),
            # }, MODEL_FILE+"."+str(epoch))
            
    torch.save(model.state_dict(), MODEL_FILE+"."+str(epoch))
    # torch.save(model.state_dict(), MODEL_FILE)
    # torch.save({
    #         'epoch': EPOCH,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': loss.item(),
    #         }, PATH+"."+str(epoch))

# 5. vaildate/test 函数
def test(dataloader, model, loss_fn, data_type="test"):
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

def estimate_ndcg_score(dataloader, model): 
    model.eval()
    with torch.no_grad():
        all_ret = []
        for _batch, (pk_date_stock,true_scores,list_label,data) in enumerate(dataloader): 
            output = model(data.to(device)) 
            
            # 准备计算分档loss，ndcg相关的数据
            ret = list(zip(pk_date_stock.tolist(), output.tolist(),true_scores.tolist(),list_label.tolist()))
            all_ret = all_ret + ret   
    
    # 计算ndcg情况
    df = pd.DataFrame(all_ret,columns=["pk_date_stock","predict","true","label"])
    evaluate_ndcg_and_scores(df)
    
def training(epoch=1):
    # 初始化
    train_data = StockPairDataset("train","highN_rate")
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    # choose,reject = next(iter(train_dataloader))
    # print(choose.shape,reject.shape)

    # validate_data = StockPairDataset("validate","highN_rate")
    # validate_dataloader = DataLoader(validate_data, batch_size=128)  
    
    # test_data = StockPairDataset("test","highN_rate")
    # test_dataloader = DataLoader(test_data, batch_size=128)  
    
    # ndcg_data = StockPointDataset(datatype="test",field="highN_rate")
    # ndcg_dataloader = DataLoader(ndcg_data, batch_size=128)  
    
    criterion = LogExpLoss() #定义损失函数
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    
    learning_rate = 0.0000001 #0.0001 #0.00001 #0.000001  #0.0000001 
    if epoch in [2,3] :
        learning_rate = 0.00001
    elif epoch == 4 :
        learning_rate = 0.00001 #000001
    else :
        learning_rate = 0.00001
    
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
    
    # epochs = 1
    # for t in range(epochs):
    print(f"Epoch={epoch}, lr={learning_rate}\n-------------------------------")
    train(train_dataloader, model, criterion, optimizer,epoch)
    # test(validate_dataloader, model, criterion, "validate")
    # test(test_dataloader, model, criterion, "test")
    # estimate_ndcg_score(ndcg_dataloader,model)
        # scheduler.step()
    
    torch.save(model.state_dict(), MODEL_FILE)
    print("Done!")

def evaluate_model_checkpoints(field = "highN_rate"):
    test_data = StockPairDataset("test",field)
    test_dataloader = DataLoader(test_data, batch_size=128)   
    
    validate_data = StockPairDataset("validate",field)
    validate_dataloader = DataLoader(validate_data, batch_size=128)  
    
    ndcg_data = StockPointDataset(datatype="test",field="highN_rate")
    ndcg_dataloader = DataLoader(ndcg_data, batch_size=128)  
    
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    criterion = LogExpLoss() #定义损失函数
    
    for i in range(13): #32
        fname =  MODEL_FILE + ".1."  + str(i) 
        print("\n" + fname)
        if os.path.isfile(fname):
            model.load_state_dict(torch.load(fname))
            test(validate_dataloader, model, criterion, "validate")
            test(test_dataloader, model, criterion, "test")
            estimate_ndcg_score(ndcg_dataloader,model)
            

def gen_date_predict_scores(model_version = "pair_high"):
    COMPARE_THRESHOLD = 0.02
    TOP_N = 10
    
    mfile = "%s.%s" %(MODEL_FILE,model_version)
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    if os.path.isfile(mfile):
        model.load_state_dict(torch.load(mfile))   
    model.eval()
    
     
    # trade_dates = load_trade_dates(conn)
    trade_dates = load_trade_dates(conn)
    for date in trade_dates:
        # print(date) 
        df = None
        data_file = "data/predict/predict_%s_%s.csv"%(date,model_version) #predict_results,predict_regress_high
        if os.path.exists(data_file):
            df = pd.read_csv(data_file, sep=",", header=0, index_col=0)
            # print(df)
        else: 
            with torch.no_grad(): 
                dataset = StockPointDataset(datatype="train",trade_date=date)
                dataloader = DataLoader(dataset, batch_size=128) 
                # print(next(iter(dataloader)))
                
                all_ = []
                for _batch, (pk_date_stock,true_scores,data) in enumerate(dataloader):         
                    output = model(data.to(device))
                    ret = list(zip(pk_date_stock.tolist(), true_scores.tolist(), output.tolist()))
                    all_ = all_ + ret
                
                df = pd.DataFrame(all_,columns=['pk_date_stock','true_score','predict_score'])
                df = df.sort_values(by=["predict_score"],ascending=False)
                df.to_csv(data_file)
        
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
        li.append(round(top_10['true_score'].mean(),3))
        
        top_5 = df.head(5)
        top_5_good =  len(top_5.loc[top_5['true_score'] > COMPARE_THRESHOLD])
        li.append(top_5_good)
        li.append(top_5_good/5)
        li.append(round(top_5['predict_score'].mean(),3))
        li.append(round(top_5['true_score'].mean(),3))
        
        top_n_5 = df.head(6).tail(5)
        top_n_5_good =  len(top_n_5.loc[top_n_5['true_score'] > COMPARE_THRESHOLD])
        li.append(top_n_5_good)
        li.append(top_n_5_good/5)
        li.append(round(top_n_5['predict_score'].mean(),3))
        li.append(round(top_n_5['true_score'].mean(),3))
        
        li.append(round(describe["true_score"]["mean"],3))
        li.append(round(describe["true_score"]["std"],3))
        li.append(round(describe["predict_score"]["mean"],3))
        li.append(round(describe["predict_score"]["std"],3))

        print( ";".join([str(item) for item in li]))
        # break 

def gen_date_predict_scores_all():
    model_files="pair_high,point_pair_high,point_high,point_low".split(",") 
    for model_name in model_files: 
        print(model_name)
        gen_date_predict_scores(model_name)

# python seq_transfomer.py training
if __name__ == "__main__": 
    op_type = sys.argv[1]
    print(op_type)
    if op_type == "training":
        epoch = int(sys.argv[2])
        training(epoch)
        # training(4)
        # training(5)
        # training(6)
        # training(7)
    if op_type == "evaluate_model_checkpoints":
        evaluate_model_checkpoints() 
    if op_type == "gen_date_predict_scores_all":
        gen_date_predict_scores_all()
    
    # python seq_transfomer.py > predict_regress_high.txt &
    # predict_results,predict_regress_high 