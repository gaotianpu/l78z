import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F

plt.plot([0,1], [1,0], '*', color='green')
plt.plot([0,1],[0,1], 'x', color='blue')

x = np.array([[0,1],[1,0],[0,0],[1,1]])
y = np.array([0,0,1,1]).reshape(-1,1) #
print(x.shape,y.shape)

data = np.array([[1, 0, 1], [0, 1, 1],
                 [1, 1, 0], [0, 0, 0]], dtype='float32')
x = data[:, :2]
y = data[:, 2]

inX = torch.Tensor(x.reshape(-1, 2))
outY = torch.Tensor(y.reshape(-1, 1))

def weight_init_normal(m):
    classname = m.__class__.__name__ #是获取类名，得到的结果classname是一个字符串
    if classname.find('Linear') != -1:  #判断这个类名中，是否包含"Linear"这个字符串，字符串的find()方法，检索这个字符串中是否包含另一个字符串
        m.weight.data.normal_(0.0, 1.)
        m.bias.data.fill_(0.)
        
# 创建一个Xor的模型
class XOrModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(XOrModel, self).__init__()
        hidden_dim = 2
        self.linear_1 = nn.Linear(input_dim, hidden_dim,bias=True)  
        self.linear_2 = nn.Linear(hidden_dim, output_dim,bias=True) 
        
        # self.fc1 = nn.Linear(2, 3)   # 隐藏层 3个神经元
        # self.fc2 = nn.Linear(3, 4)   # 隐藏层 4个神经元
        # self.fc3 = nn.Linear(4, 1)   # 输出层 1个神经元 

    def forward(self, x):
        out = torch.sigmoid(self.linear_1(x))
#         out = torch.nn.functional.relu(self.linear_1(x))
        out = self.linear_2(out)
        out = torch.sigmoid(out)
        # h1 = F.sigmoid(self.fc1(x))  # 之前也尝试过用ReLU作为激活函数, 太容易死亡ReLU了.
        # h2 = F.sigmoid(self.fc2(h1))
        # out = F.sigmoid(self.fc3(h2))
        
        return out
    
    def get_y(self,x):
        out = torch.sigmoid(self.linear_1(x))
#         out = torch.nn.functional.relu(self.linear_1(x))
        out = self.linear_2(out)
        return out
        

learning_rate = 0.1

model = XOrModel(2, 1) #模型初始化
# model.apply(weight_init_normal) #相当于net.weight_init_normal()

criterion = nn.BCELoss() #定义损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9) #定义最优化算法

# inX = torch.as_tensor(x,dtype=torch.float32) #将numpy转成tensor
# outY = torch.as_tensor(y,dtype=torch.float32).reshape(-1,1)
# print(inX.shape)
# print(outY)

# predict_Y = model(inX) #根据输入获得当前参数下的输出值
# loss = criterion(predict_Y, outY) #计算误差
# print('loss {}'.format(loss.item()))

for epoch in range(550):  #迭代次数
    optimizer.zero_grad() #清理模型里参数的梯度值
    predict_Y = model(inX) #根据输入获得当前参数下的输出值
    loss = criterion(predict_Y, outY) #计算误差
    loss.backward() #反向传播，计算梯度，
    optimizer.step() #更新模型参数
    # if epoch % 50 ==0:
    print('epoch {}, loss {}'.format(epoch, loss.item()))

# print('epoch {}, loss {}'.format(epoch, loss.item()))

# delta = 1e-7
# print("state_dict:",model.state_dict())

# plt.plot([0,1], [1,0], '*', color='green')
# plt.plot([0,1],[0,1], 'x', color='blue')


# # 怎么画出分割面，还是没搞懂，待琢磨
# model_params = list(model.parameters())
# print(model_params)
# model_weights = model_params[0].data.numpy()
# model_bias = model_params[1].data.numpy()

# x_1 = np.arange(-0.1, 1.1, 0.1)
# y_1 = ((x_1 * model_weights[0,0]) + model_bias[0]) / (-model_weights[0,1])
# plt.plot(x_1, y_1)

# x_11 = np.arange(-0.1, 1.1, 0.1)
# y_11 = ((x_11 * model_weights[1,0]) + model_bias[0]) / (-model_weights[1,1])
# plt.plot(x_11, y_11)

# # x_2 = np.arange(-0.1, 1.1, 0.1)
# # y_2 = ((x_2 * model_weights[1,0]) + model_bias[1]) / (-model_weights[1,1])
# # plt.plot(x_2, y_2)
# plt.show()


