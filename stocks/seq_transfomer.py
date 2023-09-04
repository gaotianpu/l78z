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
D_MODEL = 8  #维度


MODEL_FILE = "StockForecastModel.pth"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"  #苹果的Metal Performance Shaders（MPS）
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

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

          

# 1. 定义数据集
class StockPairDataset(Dataset):
    def __init__(self, data_type="train", field="f_high_mean_rate"):
        assert data_type in ("train", "validate", "test", "predict")
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
    def __init__(self,predict_data_file="seq_predict.txt"): 
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

class StockNDCGDataset(Dataset):
    def __init__(self,datatype="validate"): 
        dtmap = {"validate":1,"test":2}
        self.conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)
        sql = (
            "select * from stock_for_transfomer where dataset_type=%d order by trade_date desc"
            % (dtmap.get(datatype))
        )
        self.df = pd.read_sql(sql, self.conn)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pk_date_stock = self.df.iloc[idx][0]
        # date = self.df.iloc[idx][1]
        # stock_no = self.df.iloc[idx][2]
        # print(pk_date_stock,self.df.iloc[idx][4])
        data_json = json.loads(self.df.iloc[idx][4].replace("'",'"'))
        f_high_mean_rate = data_json.get("f_high_mean_rate")
        return pk_date_stock, f_high_mean_rate, torch.tensor(data_json["past_days"]) 


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
    # # 初始化
    train_data = StockPairDataset("train","f_high_mean_rate")
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    # choose,reject = next(iter(train_dataloader))
    # print(choose.shape,reject.shape)

    test_data = StockPairDataset("validate","f_high_mean_rate")
    test_dataloader = DataLoader(test_data, batch_size=128)  
    
    criterion = LogExpLoss() #定义损失函数
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    
   
    
    # train: 0.581011
    # 0.730487,0.722867,0.712951,0.708912,0.705015,0.701588 | 0.533442 | 0.501004 | 0.498912
    # layer6: 0.887414, 0.756749, 0.743146, 0.728794, 0.692854 , 0.691521 (0.690849)? , 0.690852
    # layer9: 0.819140 | 0.741584, 0.720711 | 0.705574 , 0.700156, 0.697405, 0.695821| 0.689996 
    # 0.689303？
    # 0.729, 0.691027, 0.691651
    # 0.834066, 0.703351, 0.701498, 0.699650, 0.698420, 0.697356, 0.692404, 0.691072,0.690485, 0.689984, 0.689526, 0.689427
    # 0.689188, 0.689167
    learning_rate = 0.000005 #000005  #0.0000001  
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
# 0.0000001 , 0.693031 -> 0.692935 -> 0.692743 , 5,10 epoch. norm_first = false

# 0.0000001 , 0.690696 -> 690569, 5 epoch, norm_first = true
# 0.000001, 0.690266 -> 0.689206     5 epoch, norm_first = true
# 0.00001, 0.687986 -> 0.679608  5 epoch, norm_first = true
# 0.0001, 0.653760 -> 0.609568   5 epoch, norm_first = true
# 过了 0.0001, 0.606102 -> 0.608661 开始出现不收敛
# 0.00001, 0.609200  -> 0.607962 不稳定了
# 0.000001, 0.608151 -> 0.608142


def predict():
    dataset = StockPredictDataset(predict_data_file="seq_predict.txt")
    # print(next(iter(dataset)))
    dataloader = DataLoader(dataset, batch_size=128) 
     
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)

    if os.path.isfile(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE)) 
    
    model.eval()
    with torch.no_grad():
        for _batch, (pk_date_stock,data) in enumerate(dataloader):         
            output = model(data.to(device)) 
            # print(data.size(),data.shape)
            ret = list(zip(pk_date_stock.tolist(), output.tolist()))
            print("---",_batch,"----")
            for item in ret :
                print(";".join( [str(x) for x in item])) 
            # break 

def compute_ndcg(df):
    ret = []
    date_groups = df.groupby(0)
    for date,data in date_groups:
        # print(data)
        data = data.sort_values(by=[2])
        data[4] = [math.ceil((i+1)/3) for i in range(20)]
        
        data = data.sort_values(by=[3],ascending=False)
        mean_3 = data[2].head(3).mean()
        mean_all = data[2].mean()
        
        # print(data)
        y_true = np.expand_dims(data[4].to_numpy(),axis=0)
        y_predict = np.expand_dims(data[3].to_numpy(),axis=0)
        ndcg = ndcg_score(y_true,y_predict)
        ndcg_3 = ndcg_score(y_true,y_predict,k=3)
        # print(date,ndcg)
        ret.append([date,ndcg,ndcg_3,mean_3,mean_all])
    return ret     

def estimate_ndcg_score(dataloader=None,model=None):
    if dataloader is None:
        dataset = StockNDCGDataset()
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
            # print(data.size(),data.shape)
            ret = list(zip(pk_date_stock.tolist(), f_high_mean_rate.tolist(), output.tolist()))
            
            # print("---",_batch,"----")
            li_ = li_ + [(str(item[0])[:8], str(item[0])[8:], item[1], item[2]) for item in ret]
            # break
    
    # df = pd.read_csv("ndcg.txt",sep=";", header=None)
    df = pd.DataFrame(li_)
    ret = compute_ndcg(df)
    # for x in ret:
    #     print(x)
    print("ndcg:")
    print(sum([x[1] for x in ret])/len(ret))
    print(sum([x[2] for x in ret])/len(ret))
    print(sum([x[3] for x in ret])/len(ret))
    print(sum([x[4] for x in ret])/len(ret))

def tmp():
    test_data = StockPairDataset("validate","f_high_mean_rate")
    test_dataloader = DataLoader(test_data, batch_size=128)   
    
    model = StockForecastModel(SEQUENCE_LENGTH,D_MODEL).to(device)
    criterion = LogExpLoss() #定义损失函数
    
    for i in range(10): #32
        fname =  MODEL_FILE + ".0."  + str(i) 
        print(fname)
        if os.path.isfile(fname):
            model.load_state_dict(torch.load(fname))
            test(test_dataloader, model, criterion) 


# ndcg
# 0: 0.8307093034747426,0.550872458373548 #random
# 1: 0.8486111478400458,0.5979844966625786?
# 2: 0.737837,
# 4: 0.708878, 0.8374819185466269, 0.566666881965949
# 5. 0.705350, 0.8385942260967199, 0.5700625762937057   
# 6. 0.710611, 0.8389706826370555, 0.5720949170258646 
# 7. 0.716058, 0.8464665123769485, 0.5976441339959607
# 8. 0.708704, 0.8457563130697786, 0.5908392888671835
# 9. 0.712539, 0.8452659465940661, 0.5892770887228126
# 10. 0.712425,0.8454848934851745,0.5907898843408312
# 
# 0.712466, 0.8458764051472386, 0.5915879749128049
# 0.711251, 0.84731303396431,0.5954946729416742
# train_6.data stock:date1|date2
# python seq_transfomer.py training
# python seq_transfomer.py predict
if __name__ == "__main__": 
    # estimate_ndcg_score()
    # tmp()
    op_type = sys.argv[1]
    assert op_type in ("training", "predict")
    if op_type == "predict":
        predict()
    else:
        training()
