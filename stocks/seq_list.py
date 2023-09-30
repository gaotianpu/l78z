#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import sqlite3
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

SEQUENCE_LENGTH = 20 #序列长度
D_MODEL = 9  #维度
MODEL_FILE = "StockForecastModel.list.pth"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  #苹果的Metal Performance Shaders（MPS）
    if torch.backends.mps.is_available()
    else "cpu"
)

class StockListDataset(Dataset):
    def __init__(self, data_type="train", field="f_high_mean_rate"):
        assert data_type in ("train", "validate", "test")
        self.df = pd.read_csv("list_random/%s.txt" % (data_type), sep=";", header=None)
        self.conn = sqlite3.connect("file:data/stocks_train.db?mode=ro", uri=True)
        self.field = field  # 基于哪个预测值做比较

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        str_pk_date_stock = self.df.iloc[idx][0] #.values.tolist()
        sql = (
            "select pk_date_stock,list_label,data_json from stock_for_transfomer where pk_date_stock in (%s)"
            % (str_pk_date_stock)
        )
        df_list = pd.read_sql(sql, self.conn)
        li_labels = []
        li_data = []
        for idx,row in df_list.iterrows():
            li_labels.append(float(row['list_label']))
            
            data_json = json.loads(row['data_json'])
            li_data.append(data_json['past_days']) 

        return torch.tensor(li_labels),torch.tensor(li_data)
        
# https://pytorchltr.readthedocs.io/en/stable/getting-started.html

# https://zhuanlan.zhihu.com/p/148262580

# https://www.cnblogs.com/bentuwuying/p/6690836.html


class LambdaRankLoss(nn.Module):
    """
    Get loss from one user's score output
    """
    def forward(
        self, score_predict: torch.Tensor, score_real: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param score_predict: 1xN tensor with model output score
        :param score_real: 1xN tensor with real score
        :return: Gradient of ranknet
        """
        sigma = 1.0
        score_predict_diff_mat = score_predict - score_predict.t()
        score_real_diff_mat = score_real - score_real.t()
        tij = (1.0 + torch.sign(score_real_diff_mat)) / 2.0
        lambda_ij = torch.sigmoid(sigma * score_predict_diff_mat) - tij
        loss = lambda_ij.sum(dim=1, keepdim=True) - lambda_ij.t().sum(dim=1, keepdim=True)
        return loss

# 4. train 函数
def train(dataloader, model, loss_fn, optimizer,epoch): 
    size = len(dataloader.dataset) 
    
    model.train() #训练模式
    total_loss = 0.0 
    for batch, (labels,data) in enumerate(dataloader):
        labels = torch.squeeze(labels)
        data = torch.squeeze(data)
             
        predict_scores = model(data.to(device)) 
        
        loss = loss_fn(predict_scores, labels)   
        
        # Back propagation 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
        
        total_loss = total_loss + loss.item()
        
        if batch % 64 == 0:
            avg_loss = total_loss / (batch + 1) 
            loss, current = loss.item(), (batch + 1) * len(output)
            print(f"loss: {loss:>7f} , avg_loss: {avg_loss:>7f}  [{epoch:>5d}  {current:>5d}/{size:>5d}]") 
            
        cp_save_n = 1280 #cp, checkpoint
        if batch % cp_save_n == 0:
            cp_idx = int(batch / cp_save_n)
            cp_idx_mod = cp_idx % 23
            torch.save(model.state_dict(), "%s.%s.%s" % (MODEL_FILE,epoch,cp_idx_mod) )
            
    torch.save(model.state_dict(), "%s.%s" % (MODEL_FILE,epoch))

# 5. vaildate/test 函数
def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    
    model.eval()
    with torch.no_grad():
        all_ret = []
        for _batch, (pk_date_stock,true_scores,list_label,data) in enumerate(dataloader): 
            output = model(data.to(device)) 
            
            # 准备计算分档loss，ndcg相关的数据
            ret = list(zip(pk_date_stock.tolist(), output.tolist(),true_scores.tolist(),list_label.tolist()))
            all_ret = all_ret + ret   
    
    # 计算ndcg情况
    li_ndcg = []
    df = pd.DataFrame(all_ret,columns=["pk_date_stock","predict","true","label"])
    df["trade_date"] = df.apply(lambda x: str(x['pk_date_stock'])[:8] , axis=1)
    date_groups = df.groupby('trade_date')
    for trade_date,data in date_groups: 
        y_true = np.expand_dims(data['label'].to_numpy(),axis=0)
        y_predict = np.expand_dims(data['predict'].to_numpy(),axis=0)
        ndcg = ndcg_score(y_true,y_predict)
        ndcg_5 = ndcg_score(y_true,y_predict,k=5)
        ndcg_3 = ndcg_score(y_true,y_predict,k=3)
        li_ndcg.append([ndcg,ndcg_5,ndcg_3])
        # print(trade_date,ndcg,ndcg_5,ndcg_3)
    
    ndcg_scores = [round(v,4) for v in np.mean(li_ndcg,axis=0).tolist()]
    print("ndcg_scores totoal:%s top_5:%s top_3:%s" % tuple(ndcg_scores) )  
    # df.to_csv("data/test_label_loss.txt",sep=";",index=False)
    
    

def training(field="f_high_mean_rate"):
    # 初始化
    train_data = StockListDataset(datatype="train",field=field)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    # a = next(iter(train_dataloader))
    # print(choose.shape,reject.shape)

    # vali_data = StockPointDataset(datatype="validate",field=field)
    # vali_dataloader = DataLoader(vali_data, batch_size=128)  
    
    test_data = StockPointDataset(datatype="test",field=field)
    test_dataloader = DataLoader(test_data, batch_size=128)  
    
    criterion = LambdaRankLoss() #均方差损失函数
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    
    learning_rate = 0.0000001 #0.00001 #0.000001  #0.0000001  
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=learning_rate, betas=(0.9,0.98), 
                                eps=1e-08) #定义最优化算法

    if os.path.isfile(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE)) 
        print("load success")

    epochs = 3
    start = 0
    for t in range(epochs):
        print(f"Epoch {t+start}\n-------------------------------")   
        train(train_dataloader, model, criterion, optimizer,t+start)
        # test(vali_dataloader, model, criterion)
        test(test_dataloader, model)
        # scheduler.step()
    
    torch.save(model.state_dict(), MODEL_FILE)
    print("Done!")

if __name__ == "__main__":
    op_type = sys.argv[1]
    field = sys.argv[2] #"f_low_mean_rate" # next_low_rate, f_high_mean_rate, f_low_mean_rate
    print(op_type)
    if op_type == "training":
        # python seq_list.py training f_high_mean_rate
        training(field)
    
    if op_type == "tmp":  
        # dataset = StockPairDataset('train')
        # a,b = next(iter(dataset))
        # print(a.shape,b.shape) # torch.Size([20, 9]) torch.Size([20, 9])
        # dataloader = DataLoader(dataset, batch_size=3) 
        # a,b = next(iter(dataloader))
        # print(a.shape,b.shape) # torch.Size([3, 20, 9]) torch.Size([3, 20, 9])
        
        dataset = StockListDataset('train')
        labels,data = next(iter(dataset))
        print(labels.shape,data.shape) # torch.Size([24]) torch.Size([24, 20, 9])
        
        dataloader = DataLoader(dataset, batch_size=1)
        labels,data = next(iter(dataloader))
        print(labels.shape,data.shape) #
        labels = torch.squeeze(labels)
        data = torch.squeeze(data)
        print(labels.shape,data.shape) #
        print(labels.dtype,data.dtype)
    
    