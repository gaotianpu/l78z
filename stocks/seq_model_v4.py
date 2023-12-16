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
D_MODEL = 22  #维度 

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  #苹果的Metal Performance Shaders（MPS）
    if torch.backends.mps.is_available()
    else "cpu"
)

# 1. 定义数据集
class StockPredictDataset(Dataset):
    def __init__(self,predict_data_file="seq_predict_v4.data"):
        self.df = pd.read_csv(predict_data_file, sep=";", header=None)
        self.maps = {'highN_rate':4,'next_high_rate':6,'next_low_rate':6}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pk_date_stock = int(self.df.iloc[idx][0])
        true_score = self.df.iloc[idx][4] ## field="highN_rate" data_json.get(field)
        list_label = self.df.iloc[idx][7]
        
        data_json = json.loads(self.df.iloc[idx][8])
        past_days = torch.tensor(data_json["past_days"])
        # past_days = past_days[:,:17]
        
        # 返回格式尽量和StockPointDataset保持一致
        return pk_date_stock, torch.tensor(true_score), torch.tensor(list_label), past_days

class StockPointDataset(Dataset):
    def __init__(self,datatype="validate",trade_date=None, field="highN_rate"):
        dtmap = {"train":0,"validate":1,"test":2}
        self.field = field
        self.conn = sqlite3.connect("file:data4/stocks_train_v4.db?mode=ro", uri=True)
        dataset_type = dtmap.get(datatype)
        
        if trade_date:
            sql = (
                "select pk_date_stock from stock_for_transfomer where trade_date='%s' and dataset_type=%d"
                % (trade_date,dataset_type)
            )
            # self.df = pd.read_sql(sql, self.conn)
            
            self.df = pd.read_csv(f'data4/seq_predict/f_{trade_date}.data', sep=";",header=None)
        else:
            self.df = pd.read_csv('data4/point_%s.txt' % (dataset_type), sep="|",header=None)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pk_date_stock = self.df.iloc[idx][0] 
        # print(pk_date_stock)
        sql = "select * from stock_for_transfomer where pk_date_stock=%s" % (pk_date_stock)
        df_item = pd.read_sql(sql, self.conn)
        if len(df_item)==0:
            print(pk_date_stock)
        
        data_json = json.loads(df_item.iloc[0]['data_json']) #.replace("'",'"'))
        true_score = data_json.get(self.field)
        
        past_days = torch.tensor(data_json["past_days"])
        # past_days = past_days[:,:17]
        
        list_label = df_item.iloc[0]['list_label']
        return pk_date_stock, torch.tensor(true_score), torch.tensor(list_label), past_days 


# 3. 定义模型
class StockForecastModel(nn.Module):
    def __init__(self, seq_len : int = 20, d_model: int = 22, target_dim=1) -> None:
        super().__init__()
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        self.seq_len = seq_len
        self.d_model = d_model
        self.target_dim = target_dim
        
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
    predict_data_file="seq_predict_v4.data"
    if trade_date:
        predict_data_file=f"data4/seq_predict/{trade_date}.data"
    
    print(predict_data_file)
    dataset = StockPredictDataset(predict_data_file=predict_data_file)
    dataloader = DataLoader(dataset, batch_size=128) 
    # print(next(iter(dataset)))
    
    df_merged = None 
    
    # model_ 
    order_models = "cls3".split(",")
    model_files = order_models  #+ "point_low1".split(",") #point_high1,
    
    # 2.3153
    # models_threshold_values=[0.04,0.0361,2.3153,2.3701,3.354,2.9692,1.5939,0.9008]
    # assert len(order_models)==len(models_threshold_values)
    # models_threshold = dict(zip(order_models,models_threshold_values))
    
    m = nn.Softmax(dim=1)
    m.to(device)
    
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL)
    for model_name in model_files:
        print(model_name)
        if model_name == "cls3":
            model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL,3)
        mfile = "model_v4/model_%s.pth"%(model_name)
        if os.path.isfile(mfile):
            model.load_state_dict(torch.load(mfile)) 
        
        model.to(device)
        model.eval()
        with torch.no_grad():
            all_li = []
            # pk_date_stock,true_scores,list_labels,data
            for _batch, (pk_date_stock,true_scores,list_labels,data) in enumerate(dataloader):         
                output = model(data.to(device))
                if model_name=="cls3":
                    output = m(output)
                    cls_idx = torch.argmax(output,dim=1)
                    ret = list(zip(pk_date_stock.tolist(),output[:,1].tolist(), output[:,0].tolist(), output[:,2].tolist(),cls_idx.tolist()))
                else:
                    ret = list(zip(pk_date_stock.tolist(), output.tolist()))
                all_li = all_li + ret 
                # break 
            
            columns = ["pk_date_stock",model_name]
            if model_name=="cls3":
                columns = ["pk_date_stock",model_name,model_name+"_0",model_name+"_2",model_name+"_idx"]
                
            df = pd.DataFrame(all_li,columns=columns)
            df = df.round({model_name: 4})
            # df[model_name + '_idx'] = range(len(df)) 
            # 排序,标出top5，top3
            if model_name in order_models:
                count = len(all_li)
                top3 = int(count/12)  # =count*3/24
                top5 = int(count/12) 
                
                df = df.sort_values(by=[model_name],ascending=False) #
                #[1 if i<top3 else 0 for i in range(count)]
                # df[model_name + '_top3'] = [1 if i<top3 else 0 for i in range(count)]
                # df[model_name + '_top5'] = [1 if i<top5 else 0 for i in range(count)]
                # df[model_name + '_top3'] = df.apply(
                #     lambda x: 1 if x[model_name]>models_threshold[model_name] else 0,
                #     axis=1)
            
            df.to_csv("data4/predict/predict_%s.txt"%(model_name),sep=";",index=False) 
            
            # 模型结果合并
            if df_merged is None:
                df_merged = df 
            else: # 
                df_merged = df_merged.merge(df, on="pk_date_stock",how='left')
    
    # df_merged['top3'] = df_merged[[ model_name + '_top3' for model_name in order_models ]].sum(axis=1)
    # df_merged['top5'] = df_merged[[ model_name + '_top5' for model_name in order_models ]].sum(axis=1)
                
    conn = sqlite3.connect("file:newdb/stocks.db?mode=ro", uri=True)
    trade_date = str(df_merged["pk_date_stock"][0])[:8]
    # 当天的 CLOSE_price,LOW_price,HIGH_price，作为对比基线
    df_prices = load_prices(conn,trade_date)
    df_merged = df_merged.merge(df_prices,on="pk_date_stock",how='left')
    
    # df_static_stocks = pd.read_csv("data/static_seq_stocks.txt",sep=";",header=0,dtype={'stock_no': str})
    # df_static_stocks_0 = df_static_stocks[df_static_stocks['open_rate_label']==0]
    # df_merged = df_merged.merge(df_static_stocks_0,on="stock_no",how='left')
    
    # 计算买入价和卖出价格？
    # point_low1_options = [-0.0299, -0.0158, 0, 0.0193, 0.025]
    # df_merged['buy_prices'] = ''
    # df_merged['buy_prices'] = df_merged.apply(
    #     lambda x: ','.join([str(c_round((x['point_low1'] + p + 1) * x['CLOSE_price'],2)) for p in point_low1_options]),
    #     axis=1)
    
    # point_high1_options = [-0.0266, -0.0158, 0, 0.0193, 0.0251]
    # df_merged['sell_prices'] = ''
    # df_merged['sell_prices'] = df_merged.apply(
    #     lambda x:','.join([str(c_round((x['point_high1'] + p + 1) * x['CLOSE_price'],2)) for p in point_high1_options]),
    #     axis=1)
    
    # point_high1 效果更好些？
    df_merged = df_merged.sort_values(by=["cls3"],ascending=False) # 
    df_merged.to_csv("data4/predict/predict_merged.txt.%s"%(trade_date),sep=";",index=False) 
    df_merged.to_csv("data4/predict/predict_merged.txt",sep=";",index=False) 
    
    # 暂时先不关注科创板
    df_merged = df_merged[ (df_merged['stock_no'].str.startswith('688') == False)]
    
    sel_fields = "pk_date_stock,stock_no,top3,point_high,point_high1,pair_date,point_low1,CLOSE_price,LOW_price,HIGH_price,buy_prices,sell_prices".split(",")
    sel_fields = "pk_date_stock,stock_no,cls3,cls3_0,cls3_2,cls3_idx,CLOSE_price,LOW_price,HIGH_price".split(",")
    df_merged[sel_fields].to_csv("predict_merged_for_show_v4.txt",sep=";",index=False) 

def predict_many():
    start_date=20231017
    conn = sqlite3.connect("file:newdb/stocks.db?mode=ro", uri=True)
    sql = f"select distinct trade_date from stock_raw_daily where trade_date>{start_date}"
    df = pd.read_sql(sql, conn)
    trade_dates = df['trade_date'].sort_values(ascending=True).tolist()
    for idx,trade_date in enumerate(trade_dates):
        print(trade_date) 
        predict(trade_date)


        
    

def evaluate_models(dataset_type="test",field="highN_rate"):
    t_data = StockPointDataset(dataset_type,field=field)
    t_dataloader = DataLoader(t_data, batch_size=128)
    
    order_models = "point_high,point_high_label9,point_high1,pair_date,pair_date_r0,pair_date_r1,pair_date_r2,list,list_1".split(",")
    model_files = order_models + "point_low1".split(",") #point_high1,
    
    df_merged = None 
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL)
    for model_name in model_files:
        print(model_name,dataset_type)
        
        df = None
        model_output_file = f"data4/evaluate/{dataset_type}_{model_name}.txt"
        if os.path.exists(model_output_file):
            df = pd.read_csv(model_output_file,sep=";",header=0,dtype={'pk_date_stock': int})
        else:
            mfile = "model/model_%s.pth"%(model_name)
            if os.path.isfile(mfile):
                model.load_state_dict(torch.load(mfile))
            
            model.to(device)
            model.eval()
            with torch.no_grad():
                all_li = []
                for _batch, (pk_date_stock,true_scores,list_label,data) in enumerate(t_dataloader):         
                    output = model(data.to(device))
                    ret = list(zip(pk_date_stock.tolist(), output.tolist(),true_scores.tolist(),list_label.tolist()))
                    all_li = all_li + ret 
                    # break
                #valid=481642, test=505344
                df = pd.DataFrame(all_li,columns=["pk_date_stock","true","label",model_name])
                df = df.round({model_name: 4})
                df = df.sort_values(by=[model_name],ascending=False)
                df.to_csv(model_output_file,sep=";",index=False) 
        
        # total_cnt = len(df)
        # for pct in [10,15,20,40,50,80,100,200,250,300]:
        #     topN = int(total_cnt/pct)
        #     true_mean = df.head(topN)["true"].mean()
        #     print(model_name,pct,df.iloc[topN][model_name],true_mean)
        # topN = int(total_cnt/250)
        # df[model_name + '_top3'] = [1 if i<topN else 0 for i in range(total_cnt)]  
                        
        if df_merged is None:
            df_merged = df
        else:
            tmp_df = df[["pk_date_stock",model_name]]
            df_merged = df_merged.merge(tmp_df,on="pk_date_stock",how='left')
            # print(df_merged.columns)
    
    # print(df_merged.columns)
    # print("top3")
    # df_merged['top3'] = df_merged[[ model_name + '_top3' for model_name in order_models ]].sum(axis=1)   
    # df_merged = df_merged.sort_values(by=["top3"],ascending=False)  
    # for pct in [10,15,20,40,50,80,100,200,250]:
    #     topN = int(total_cnt/pct)
    #     true_mean = df_merged.head(topN)["true"].mean()
    #     print("top3",pct,df_merged.iloc[topN]["top3"],true_mean)
    
    # df_merged['top3_1'] = df_merged.apply(
    #             lambda x: 1 if x['top3']>0 else 0,
    #             axis=1)
    
    # [1 if df_merged['top3']>0 else 0] #[df_merged['top3']>0]
    # df_merged =  df_merged.sort_values(by=["top3_1","point_high"],ascending=False)   
    # print("final",len(df_merged[df_merged['top3_1']==1]))
    # for pct in [10,15,20,40,50,80,100,200,250]:
    #     topN = int(total_cnt/pct)
    #     true_mean = df_merged.head(topN)["true"].mean()
    #     print("final",pct,df_merged.iloc[topN]["top3"],true_mean) 
    
    df_merged.to_csv(f"data4/evaluate/{dataset_type}_merged.txt",sep=";",index=False) 
    
def compute_prices(dataset_type):
    conn = sqlite3.connect("file:data4/stocks_train_v4.db?mode=ro", uri=True)
    
    df = pd.read_csv(f"data4/evaluate/{dataset_type}_merged.txt",sep=";",dtype={'pk_date_stock':int})
    for idx,row in df.iterrows():   
        pk_date_stock = int(row['pk_date_stock'])
        # print(pk_date_stock)  #,high2_rate
        sql = f"select pk_date_stock,low1_rate,high2_rate from stock_for_transfomer where pk_date_stock={pk_date_stock};" 
        df_items = pd.read_sql(sql, conn)
        if len(df_items)==0:
            print("not exist")
            continue
        print(pk_date_stock,df_items['low1_rate'][0],df_items['high2_rate'][0])
        break 
        
    
if __name__ == "__main__":
    op_type = sys.argv[1]
    if op_type == "predict":
        # python seq_model_v4.py predict
        predict()
    if op_type == "many":
        predict_many()
    if op_type == "evaluate_models":
        evaluate_models("validate")
        evaluate_models("test")
        # compute_prices("validate")
        # compute_prices("test")
        