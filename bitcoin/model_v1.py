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

SEQUENCE_LENGTH = 20 #序列长度
D_MODEL = 24  #维度 

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  #苹果的Metal Performance Shaders（MPS）
    if torch.backends.mps.is_available()
    else "cpu"
)

# 1. 定义数据集
class BtcPointDataset(Dataset):
    def __init__(self,datatype="train",field="highN_rate"):
        self.field = field
        dtmap = {"train":0,"validate":1,"test":2,"predict":3}
        dataset_type = dtmap.get(datatype)
        self.conn = sqlite3.connect("file:data/btc_train.db?mode=ro", uri=True)
        
        #预测不同的字段，可能需要不同的数据清洗逻辑，每种类型单独构建数据文件
        self.df = pd.read_csv(f'data/point_{dataset_type}_{field}.csv', 
                              sep=";",header=None,names=['pk_date_btc',field]) 
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pk_date_btc = self.df.iloc[idx][0] 
        
        sql = f"select * from train_data where pk_date_btc={pk_date_btc}"
        df_item = pd.read_sql(sql, self.conn)
        if len(df_item)==0:
            print(pk_date_btc)
        
        data_json = json.loads(df_item.iloc[0]['data_json'])
        true_score = torch.tensor(data_json.get(self.field))
        past_days = torch.tensor(data_json["past_days"])
        
        return pk_date_btc, true_score, past_days 


# 3. 定义模型
class BtcForecastModel(nn.Module):
    def __init__(self, seq_len : int = 20, d_model: int = 24, target_dim=1) -> None:
        super().__init__()
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        self.seq_len = seq_len
        self.d_model = d_model
        self.target_dim = target_dim
        
        nhead = 1 #1,2？
        num_layers = 3 # 6,
        dim_feedforward = d_model * num_layers #是否合理？
        
        self.position_embedding = nn.Embedding(self.seq_len, self.d_model)
        
        # activation = "gelu"
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            activation = "gelu", batch_first = True, norm_first = True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.value_head = nn.Linear(self.d_model, self.target_dim)

    def forward(
        self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor: 
        pos = torch.arange(0, self.seq_len, dtype=torch.long).unsqueeze(0).to(device)
        pos_emb = self.position_embedding(pos)
        
        out = self.transformer_encoder(sequences + pos_emb)
        values = self.value_head(out) 
        
        value = values.mean(dim=1).squeeze(1)  # ensure shape is (B)
        return value

def test():
    dataset = BtcPointDataset(datatype="train",field="highN_rate")
    pk_date_btc, true_score, past_days = next(iter(dataset))
    print(pk_date_btc, true_score, past_days) 
    
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # model = BtcForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    # model.train() #训练模式
    # for batch, (pk_date_stock,true_scores,data) in enumerate(dataloader):         
    #     output = model(data.to(device))
    #     print(output)
    #     break
    
if __name__ == "__main__":
    test()