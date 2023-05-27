import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F

# from xor import XOrModel

class XOrModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(XOrModel, self).__init__()
        hidden_dim = 2
        self.fc1 = nn.Linear(input_dim, hidden_dim,bias=True)  
        self.fc2 = nn.Linear(hidden_dim, output_dim,bias=True)  

    def forward(self, x):
        out = torch.sigmoid(self.fc1(x))  
        out = self.fc2(out) 
        out = torch.sigmoid(out)
        return out 
    
    def output(self,x):
        out = torch.sigmoid(self.fc1(x)) 
        print("first:",x,out)
        
        out1 = self.fc2(out) 
        out1 = torch.sigmoid(out1)  
        print("second:",out1)
        return out,out1 

plt.plot([0,1], [1,0], '*', color='green')
plt.plot([0,1],[0,1], 'x', color='blue')

x = np.array([[0,1],[1,0],[0,0],[1,1]])
y = np.array([0,0,1,1]).reshape(-1,1) #
# print(x.shape,y.shape)
# plt.show()

inX = torch.Tensor(x.reshape(-1, 2))
outY = torch.Tensor(y.reshape(-1, 1))


model = torch.load('xor.pt')
model.output(inX)

# 画出 第一层的两个维度的独立投影，经第一层sigmod后的位置投影，
model_params = list(model.parameters())
model_weights = model_params[0].data.numpy()
model_bias = model_params[1].data.numpy()

model_weights1 = model_params[2].data.numpy()
model_bias1 = model_params[3].data.numpy()

# 第一层的两个维度的投影函数
x_1 = np.arange(-0.1, 1.1, 0.1)
y_1 = ((x_1 * model_weights[0,0]) + model_bias[0]) / (-model_weights[0,1])
plt.plot(x_1, y_1, color='green')

x_11 = np.arange(-0.1, 1.1, 0.1)
y_11 = ((x_11 * model_weights[1,0]) + model_bias[1]) / (-model_weights[1,1])
plt.plot(x_11, y_11, color='blue')

# 经第一层投影+sigmod后原始输入投影到的新位置
out = model.output(inX)
tmp = out[0].detach().numpy()
plt.plot(tmp[0],tmp[1], '*', color='black')
plt.plot(tmp[2],tmp[3], 'x', color='orange')

# 基于投影后的位置，执行分割
x_2 = np.arange(-0.1, 1.1, 0.1)
y_2 = ((x_2 * model_weights1[0,0]) + model_bias1[0]) / (-model_weights1[0,1])
plt.plot(x_2, y_2, color='red')
plt.show()
