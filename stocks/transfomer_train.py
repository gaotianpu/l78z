import os
import sys
from typing import Optional
import numpy as np
import pandas as pd
import json
import sqlite3
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

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
    def __init__(self, data_type="train", field="f_mean_rate"):
        assert data_type in ("train", "validate", "test", "predict")
        self.df = pd.read_csv("%s.data" % (data_type), sep=" ", header=None)
        self.conn = sqlite3.connect("file:data/stocks.db?mode=ro", uri=True)
        self.field = field  # 基于哪个预测值做比较

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sql = (
            "select pk_date_stock,data_json from stock_for_transfomer_test where pk_date_stock in (%s,%s)"
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

# 3. 定义模型
class StockForecastModel(nn.Module):
    def __init__(self, seq_len : int = 20, d_model: int = 8) -> None:
        super().__init__()
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
        # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
        self.seq_len = seq_len
        self.d_model = d_model
        
        nhead = 2
        num_layers = 6 
        dim_feedforward = d_model * num_layers #是否合理？
        
        self.position_embedding = nn.Embedding(self.seq_len, self.d_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=2, dim_feedforward=dim_feedforward, norm_first = True
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
    for batch, (choose,reject) in enumerate(dataloader):         
        c = model(choose.to(device))
        r = model(reject.to(device))  
        
        loss = loss_fn(c, r)   
        
        # Back propagation 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 
        
        if batch % 64 == 0:
            loss, current = loss.item(), (batch + 1) * len(choose)
            print(f"loss: {loss:>7f}  [{epoch:>5d}  {current:>5d}/{size:>5d}]") 
        
        if batch % 512 == 0:
            torch.save(model.state_dict(), MODEL_FILE+"."+str(epoch))
            # torch.save({
            # 'epoch': epoch,
            # 'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss.item(),
            # }, MODEL_FILE+"."+str(epoch))
            
    torch.save(model.state_dict(), MODEL_FILE+"."+str(epoch))
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

# # 初始化
train_data = StockPairDataset("train","f_mean_rate")
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
# choose,reject = next(iter(train_dataloader))
# print(choose.shape,reject.shape)

test_data = StockPairDataset("validate","f_mean_rate")
test_dataloader = DataLoader(test_data, batch_size=128)

seq_len = 20
d_model = 8 

learning_rate = 0.0000001

criterion = LogExpLoss() #定义损失函数
model = StockForecastModel(seq_len,d_model).to(device)
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, betas=(0.9,0.98), 
                             eps=1e-08) #定义最优化算法
# scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

if os.path.isfile(MODEL_FILE):
    # model.load_state_dict(torch.load(MODEL_FILE)) 
    # checkpoint = torch.load(MODEL_FILE)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    print("load success")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, criterion, optimizer,t)
    test(test_dataloader, model, criterion)
    # scheduler.step()
    
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
# dataset.conn.close()
