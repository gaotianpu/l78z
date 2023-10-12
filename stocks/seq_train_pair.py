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
from seq_model import StockForecastModel,StockPointDataset,SEQUENCE_LENGTH,D_MODEL,evaluate_ndcg_and_scores

# SEQUENCE_LENGTH = 20 #序列长度
# D_MODEL = 9  #维度
MODEL_FILE = "StockForecastModel.pair.pth"

conn = sqlite3.connect("file:data/stocks_train.db?mode=ro", uri=True)

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

class StockPairDataset(Dataset):
    def __init__(self, data_type="train", field="f_high_mean_rate"):
        assert data_type in ("train", "validate", "test")
        self.df = pd.read_csv("%s.data" % (data_type), sep=";", header=None)
        self.conn = sqlite3.connect("file:data/stocks_train.db?mode=ro", uri=True)
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
            print(f"loss: {loss:>7f} , avg_loss: {avg_loss:>7f}  [{epoch:>5d}  {current:>5d}/{size:>5d} {rate}%]]") 
        
        if batch % 512 == 0:
            torch.save(model.state_dict(), MODEL_FILE+"."+str(epoch) + "." + str(int(batch / 512)) )
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
    print(f"{data_type} Avg loss: {test_loss:>8f} \n")

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
    
def training():
    # 初始化
    train_data = StockPairDataset("train","f_high_mean_rate")
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    # choose,reject = next(iter(train_dataloader))
    # print(choose.shape,reject.shape)

    validate_data = StockPairDataset("validate","f_high_mean_rate")
    validate_dataloader = DataLoader(validate_data, batch_size=128)  
    
    test_data = StockPairDataset("test","f_high_mean_rate")
    test_dataloader = DataLoader(test_data, batch_size=128)  
    
    ndcg_data = StockPointDataset(datatype="test",field="f_high_mean_rate")
    ndcg_dataloader = DataLoader(ndcg_data, batch_size=128)  
    
    criterion = LogExpLoss() #定义损失函数
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    
    learning_rate = 0.00001 #0.0001 #0.00001 #0.000001  #0.0000001  
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
        train(train_dataloader, model, criterion, optimizer,t+1)
        test(validate_dataloader, model, criterion, "validate")
        test(test_dataloader, model, criterion, "test")
        estimate_ndcg_score(ndcg_dataloader,model)
        # scheduler.step()
    
    torch.save(model.state_dict(), MODEL_FILE)
    print("Done!")

def evaluate_model_checkpoints():
    test_data = StockPairDataset("validate","f_high_mean_rate")
    test_dataloader = DataLoader(test_data, batch_size=128)   
    
    ndcg_data = StockPointDataset(datatype="test",field="f_high_mean_rate")
    ndcg_dataloader = DataLoader(ndcg_data, batch_size=128)  
    
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    criterion = LogExpLoss() #定义损失函数
    
    for i in range(31): #32
        fname =  MODEL_FILE + ".0."  + str(i) 
        print(fname)
        if os.path.isfile(fname):
            model.load_state_dict(torch.load(fname))
            test(test_dataloader, model, criterion) 
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
        training()
    if op_type == "evaluate_model_checkpoints":
        evaluate_model_checkpoints() 
    if op_type == "gen_date_predict_scores_all":
        gen_date_predict_scores_all()
    
    # python seq_transfomer.py > predict_regress_high.txt &
    # predict_results,predict_regress_high 