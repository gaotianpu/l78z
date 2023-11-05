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

from common import load_prices,c_round

SEQUENCE_LENGTH = 20 #序列长度
D_MODEL = 29  #维度 

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  #苹果的Metal Performance Shaders（MPS）
    if torch.backends.mps.is_available()
    else "cpu"
)

# 1. 定义数据集
class StockPredictDataset(Dataset):
    def __init__(self,predict_data_file="seq_predict_v2.data"): 
        self.df = pd.read_csv(predict_data_file, sep=";", header=None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pk_date_stock = self.df.iloc[idx][0]
        # date = self.df.iloc[idx][1]
        # stock_no = self.df.iloc[idx][2]
        # print(pk_date_stock,self.df.iloc[idx][5])
        data_json = json.loads(self.df.iloc[idx][5].replace("'",'"'))
        return pk_date_stock, torch.tensor(data_json["past_days"]) 

class StockPointDataset(Dataset):
    def __init__(self,datatype="validate",trade_date=None, field="f_high_mean_rate"): 
        dtmap = {"train":0,"validate":1,"test":2}
        self.field = field
        self.conn = sqlite3.connect("file:data/stocks_train_3.db?mode=ro", uri=True)
        dataset_type = dtmap.get(datatype)
        
        if trade_date:
            sql = (
                "select pk_date_stock from stock_for_transfomer where trade_date='%s' and dataset_type=%d"
                % (trade_date,dataset_type)
            )
            self.df = pd.read_sql(sql, self.conn)
        else:
            self.df = pd.read_csv('data2/point_%s.txt' % (dataset_type), header=None)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pk_date_stock = self.df.iloc[idx][0] 
        sql = "select * from stock_for_transfomer where pk_date_stock=%s" % (pk_date_stock)
        df_item = pd.read_sql(sql, self.conn) 
        
        data_json = json.loads(df_item.iloc[0]['data_json']) #.replace("'",'"'))
        true_score = data_json.get(self.field)
        
        past_days = torch.tensor(data_json["past_days"])
        # past_days = past_days[:,:17]
        
        list_label = df_item.iloc[0]['list_label']
        return pk_date_stock, torch.tensor(true_score), torch.tensor(list_label), past_days 


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

def evaluate_ndcg_and_scores(df):
    # df = pd.DataFrame(all_ret,columns=["pk_date_stock","predict","true","label"])
    # 计算ndcg情况
    df["trade_date"] = df.apply(lambda x: str(x['pk_date_stock'])[:8] , axis=1)
    
    li_ndcg = []
    date_groups = df.groupby('trade_date')
    for trade_date,data in date_groups: 
        y_true = np.expand_dims(data['label'].to_numpy(),axis=0)
        y_predict = np.expand_dims(data['predict'].to_numpy(),axis=0)
        ndcg = ndcg_score(y_true,y_predict)
        ndcg_5 = ndcg_score(y_true,y_predict,k=5)
        ndcg_3 = ndcg_score(y_true,y_predict,k=3)
        
        # 计算实际收益率，涨停无法买入的排除？
        idx = np.argsort(y_predict,axis=1)
        y_true_scores = np.expand_dims(data['true'].to_numpy(),axis=0)
        y_true_sorted = np.take(y_true_scores,idx).squeeze()
        ta = y_true_sorted.mean()
        t3 = y_true_sorted[-3:].mean()
        t5 = y_true_sorted[-5:].mean()
        
        li_ndcg.append([ndcg,ndcg_5,ndcg_3,ta,t5,t3])
    
    ndcg_scores = [round(v,4) for v in np.mean(li_ndcg,axis=0).tolist()]
    print("ndcg_scores:n=%s,n5=%s,n3=%s , true_rate:t=%s,t5=%s,t3=%s" % tuple(ndcg_scores) )  
    return df
   
       
def predict(trade_date=None):
    predict_data_file="seq_predict_v2.data"
    if trade_date:
        predict_data_file=f"data/seq_predict_v2/{trade_date}.data"
    
    print(predict_data_file)
    dataset = StockPredictDataset(predict_data_file=predict_data_file)
    dataloader = DataLoader(dataset, batch_size=128) 
    # print(next(iter(dataset)))
    
    df_merged = None 
    
    # model_v3 model_v3/model_point2pair_dates.pth
    # list_stocks, ,pair_dates,pair_dates_stocks
    order_models = "list_dates,point,point2pair_dates,point_high1".split(",")
    model_files = order_models + "point_low,point_low1".split(",") #point_high1,
    
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    for model_name in model_files:
        print(model_name)
        mfile = "model_v3/model_%s.pth"%(model_name)
        if os.path.isfile(mfile):
            model.load_state_dict(torch.load(mfile)) 
        
        model.eval()
        with torch.no_grad():
            all_li = []
            for _batch, (pk_date_stock,data) in enumerate(dataloader):         
                output = model(data.to(device))
                ret = list(zip(pk_date_stock.tolist(), output.tolist()))
                all_li = all_li + ret 
                # break 
            
            df = pd.DataFrame(all_li,columns=["pk_date_stock",model_name])
            df = df.round({model_name: 4})
            # df[model_name + '_idx'] = range(len(df)) 
            # 排序,标出top5，top3
            if model_name in order_models:
                count = len(all_li)
                top3 = int(count/8)  # =count*3/24
                top5 = int(count*5/24) 
                
                df = df.sort_values(by=[model_name],ascending=False) # 
                df[model_name + '_top3'] = [1 if i<top3 else 0 for i in range(count)]
                df[model_name + '_top5'] = [1 if i<top5 else 0 for i in range(count)]
            
            df.to_csv("data/predict_v2/predict_%s.txt"%(model_name),sep=";",index=False) 
            
            # 模型结果合并
            if df_merged is None:
                df_merged = df 
            else: # 
                df_merged = df_merged.merge(df, on="pk_date_stock",how='left')
    
    df_merged['top3'] = df_merged[[ model_name + '_top3' for model_name in order_models ]].sum(axis=1)
    df_merged['top5'] = df_merged[[ model_name + '_top5' for model_name in order_models ]].sum(axis=1)
                
    conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)
    trade_date = str(df_merged["pk_date_stock"][0])[:8]
    # 当天的 CLOSE_price,LOW_price,HIGH_price，作为对比基线
    df_prices = load_prices(conn,trade_date)
    df_merged = df_merged.merge(df_prices,on="pk_date_stock",how='left')
    
    df_static_stocks = pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0,dtype={'stock_no': str})
    df_static_stocks_0 = df_static_stocks[df_static_stocks['open_rate_label']==0]
    df_merged = df_merged.merge(df_static_stocks_0,on="stock_no",how='left')
    
    df_merged.to_csv("data/predict_v2/predict_merged_middle_tmp.txt.%s"%(trade_date),sep=";",index=False) 
    
    point_low1_options = [-0.0299, -0.0158, 0, 0.0193, 0.025]
    for i,p in  enumerate(point_low1_options):
        df_merged[f'point_low1_{i}'] = c_round(df_merged['point_low1'] + p)
    
    point_high1_options = [-0.0266, -0.0158, 0, 0.0193, 0.0251]
    for i,p in  enumerate(point_high1_options):
        df_merged[f'point_high_{i}'] = c_round(df_merged['point_high1'] + p)
  
    df_merged['buy_prices'] = ''
    df_merged['sell_prices'] = ''
    
    # 计算买入价和卖出价格？
    for idx,row in df_merged.iterrows():
        low_rates = [row['point_low1'] + p for p in point_low1_options]
        buy_prices = (np.array(sorted(low_rates))+1) * row['CLOSE_price']
        df_merged.loc[idx, 'buy_prices'] = ','.join([str(v) for v in buy_prices.round(2)]) 
        
        high_rates = [row['point_high1'] + p for p in point_high1_options]
        sell_prices = (np.array(sorted(high_rates))+1) * row['CLOSE_price']
        df_merged.loc[idx, 'sell_prices'] = ','.join([str(v) for v in sell_prices.round(2)])
    
    # point_high1 效果更好些？
    df_merged = df_merged.sort_values(by=["top3","point_high1"],ascending=False) # 
    df_merged.to_csv("data/predict_v2/predict_merged.txt.%s"%(trade_date),sep=";",index=False) 
    df_merged.to_csv("data/predict_v2/predict_merged.txt",sep=";",index=False) 
    
    # 暂时先不关注科创板
    df_merged = df_merged[ (df_merged['stock_no'].str.startswith('688') == False)]
    
    sel_fields = "pk_date_stock,stock_no,top3,point_high1,point2pair_dates,point,point_low,point_low1,CLOSE_price,LOW_price,HIGH_price,buy_prices,sell_prices".split(",")
    df_merged[sel_fields].to_csv("predict_merged_for_show_v2.txt",sep=";",index=False) 

def predict_many():
    start_date=20231017
    conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)
    sql = f"select distinct trade_date from stock_raw_daily_2 where trade_date>{start_date}"
    df = pd.read_sql(sql, conn)
    trade_dates = df['trade_date'].sort_values(ascending=True).tolist()
    for idx,trade_date in enumerate(trade_dates):
        print(trade_date) 
        predict(trade_date)
    
if __name__ == "__main__":
    op_type = sys.argv[1]
    if op_type == "predict":
        # python seq_model_v2.py predict
        predict()
    if op_type == "many":
        predict_many()
        