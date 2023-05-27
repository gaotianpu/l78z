import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F

plt.plot([0,1], [1,0], '*', color='green')
plt.plot([0,1],[0,1], 'x', color='blue')



x = np.array([[0,1],[1,0],[0,0],[1,1]])
y = np.array([0,0,1,1]).reshape(-1,1) #
# print(x.shape,y.shape)
# plt.show()

inX = torch.Tensor(x.reshape(-1, 2))
outY = torch.Tensor(y.reshape(-1, 1))
        
# 创建一个Xor的模型
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
        

model = XOrModel(2, 1) #模型初始化
criterion = nn.BCELoss() #定义损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9) #定义优化算法

# 开始训练
for epoch in range(550):  #迭代次数
    optimizer.zero_grad() #清理模型里参数的梯度值
    predict_Y = model(inX) #根据输入获得当前参数下的输出值
    loss = criterion(predict_Y, outY) #计算误差
    loss.backward() #反向传播，计算梯度，
    optimizer.step() #更新模型参数
    # if epoch % 50 ==0:
    print('epoch {}, loss {}'.format(epoch, loss.item()))

torch.save(model,'xor.pt')
out = model.output(inX)
# print(out[0])
# print(out[1])
tmp = out[0].detach().numpy()
print(tmp[0][0],tmp[0][1])
print(tmp[1][0],tmp[1][1])
print(tmp[2][0],tmp[2][1])
print(tmp[3][0],tmp[3][1])

# plt.plot([tmp[0][0],tmp[0][1]], [tmp[1][0],tmp[1][1]], '*', color='black')
# plt.plot([tmp[2][0],tmp[2][1]], [tmp[3][0],tmp[3][1]], 'x', color='orange')

plt.plot(tmp[0],tmp[1], '*', color='black')
plt.plot(tmp[2],tmp[3], 'x', color='orange')


# plt.plot([0.015265385,0.94249374], [0.015389745,0.94256026], '*', color='black')
# plt.plot([0.9569296,0.99979144],[0.000010905584,0.0531352], 'x', color='orange')

# # 画出分割面
model_params = list(model.parameters())
model_weights = model_params[0].data.numpy()
model_bias = model_params[1].data.numpy()

model_weights1 = model_params[2].data.numpy()
model_bias1 = model_params[3].data.numpy()

x_1 = np.arange(-0.1, 1.1, 0.1)
y_1 = ((x_1 * model_weights[0,0]) + model_bias[0]) / (-model_weights[0,1])
# y_1 = F.sigmoid(torch.Tensor(y_1)).numpy()
plt.plot(x_1, y_1, color='green')

x_11 = np.arange(-0.1, 1.1, 0.1)
y_11 = ((x_11 * model_weights[1,0]) + model_bias[1]) / (-model_weights[1,1])
# y_11 = F.sigmoid(torch.Tensor(y_11)).numpy()
plt.plot(x_11, y_11, color='blue')

x_2 = np.arange(-0.1, 1.1, 0.1)
y_2 = ((x_2 * model_weights1[0,0]) + model_bias1[0]) / (-model_weights1[0,1])
plt.plot(x_2, y_2, color='red')
plt.show()


