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

from common import load_trade_dates,load_prices

conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)

SEQUENCE_LENGTH = 20 #序列长度
D_MODEL = 9  #维度
MODEL_FILE = "StockForecastModel.pth" 

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  #苹果的Metal Performance Shaders（MPS）
    if torch.backends.mps.is_available()
    else "cpu"
)

# 1. 定义数据集
class StockPairDataset(Dataset):
    def __init__(self, data_type="train", field="f_high_mean_rate"):
        assert data_type in ("train", "validate", "test")
        self.df = pd.read_csv("%s.data" % (data_type), sep=" ", header=None)
        self.conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)
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

class StockPredictDataset(Dataset):
    def __init__(self,predict_data_file="seq_predict.data"): 
        self.df = pd.read_csv(predict_data_file, sep=";", header=None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pk_date_stock = self.df.iloc[idx][0]
        # date = self.df.iloc[idx][1]
        # stock_no = self.df.iloc[idx][2]
        # print(pk_date_stock,self.df.iloc[idx][4])
        data_json = json.loads(self.df.iloc[idx][4].replace("'",'"'))
        return pk_date_stock, torch.tensor(data_json["past_days"]) 

class StockPointDataset(Dataset):
    def __init__(self,datatype="validate",trade_date=None, field="f_high_mean_rate"): 
        dtmap = {"train":0,"validate":1,"test":2}
        self.field = field
        self.conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)
        # order by trade_date desc
        sql = (
            "select pk_date_stock from stock_for_transfomer where dataset_type=%d"
            % (dtmap.get(datatype))
        )
        # #debug
        # sql = (
        #     "select pk_date_stock from stock_for_transfomer where dataset_type>=%d"
        #     % (dtmap.get(datatype))
        # )
        
        if trade_date:
            sql = (
                "select pk_date_stock from stock_for_transfomer where trade_date='%s' and dataset_type=%d"
                % (trade_date,dtmap.get(datatype))
            )
            
        self.df = pd.read_sql(sql, self.conn)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pk_date_stock = self.df.iloc[idx][0]
        # data_json = json.loads(self.df.iloc[idx][4].replace("'",'"'))
        
        sql = "select * from stock_for_transfomer where pk_date_stock=%s" % (pk_date_stock)
        df_item = pd.read_sql(sql, self.conn) 
        
        data_json = json.loads(df_item.iloc[0][4].replace("'",'"'))
        true_score = data_json.get(self.field)
        return pk_date_stock, torch.tensor(true_score), torch.tensor(data_json["past_days"]) 


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

# 3. 定义模型
class StockForecastModel(nn.Module):
    def __init__(self, seq_len : int = 20, d_model: int = 9) -> None:
        super().__init__()
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        self.seq_len = seq_len
        self.d_model = d_model
        
        nhead = 1 #1,2？
        num_layers = 9 # 6,
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
        self.value_head = nn.Linear(self.d_model, 1)

    def forward(
        self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor: 
        pos = torch.arange(0, self.seq_len, dtype=torch.long).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        
        out = self.transformer_encoder(sequences + pos_emb)
        values = self.value_head(out)
        value = values.mean(dim=1).squeeze(1)  # ensure shape is (B)
        return value

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model) #[seq_len, batch_size, embedding_dim]
#         #pe = torch.zeros(1, max_len, d_model) #[batch_size, seq_len, embedding_dim]
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)

def predict():
    dataset = StockPredictDataset(predict_data_file="seq_predict.data")
    dataloader = DataLoader(dataset, batch_size=128) 
    # print(next(iter(dataset)))
    
    model_files="pair_high,point_pair_high,point_high,point_low".split(",") 
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    li_df = []
    for model_name in model_files:
        mfile = "StockForecastModel.pth.%s"%(model_name)
        if os.path.isfile(mfile):
            model.load_state_dict(torch.load(mfile)) 
        
        model.eval()
        with torch.no_grad():
            all = []
            for _batch, (pk_date_stock,data) in enumerate(dataloader):         
                output = model(data.to(device))
                ret = list(zip(pk_date_stock.tolist(), output.tolist()))
                all = all + ret 
                # break 
            
            df = pd.DataFrame(all,columns=["pk_date_stock",model_name])
            df = df.sort_values(by=[model_name],ascending=False)
            df = df.round(4)
            df['idx_' + model_name] = range(len(df))
            df.to_csv("data/predict_%s.txt"%(model_name),sep=";",index=False)
            li_df.append(df)
            
    # 几个模型合并
    trade_date = str(li_df[0]["pk_date_stock"][0])[:8]
    
    df = li_df[0].merge(li_df[1],on="pk_date_stock",how='left')
    df = df.merge(li_df[2],on="pk_date_stock",how='left')
    df = df.merge(li_df[3],on="pk_date_stock",how='left')
    df['idx_merge'] = df.apply(lambda x: x['idx_pair_high'] + x['idx_point_pair_high'], axis=1)
    
    df_prices = load_prices(conn,trade_date)
    df = df.merge(df_prices,on="pk_date_stock",how='left')
    df['buy_low_price'] = df.apply(lambda x: x['CLOSE_price'] * (1 + x['point_low']), axis=1)
    df['buy_low_price'] = df['buy_low_price'].round(2)
    
    #df["buy_low_price"] = df_prices['CLOSE_price'].values * (1+df_prices['point_low'].values)
    
    df.to_csv("data/predict_merged.txt.%s"%(trade_date),sep=";",index=False) 
    df.to_csv("data/predict_merged.txt",sep=";",index=False) 

if __name__ == "__main__":
    op_type = sys.argv[1]
    assert op_type in ("training", "predict")
    if op_type == "predict":
        predict()
    
    # dataset = StockPointDataset(datatype="validate")
    # d = next(iter(dataset))
    # print(d)