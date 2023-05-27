# 利用Pytorch解决XOR问题
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

data = np.array([[1, 0, 1], [0, 1, 1],
                 [1, 1, 0], [0, 0, 0]], dtype='float32')
x = data[:, :2]
y = data[:, 2]


# 初始化权重变量
def weight_init_normal(m):
    classname = m.__class__.__name__ #是获取类名，得到的结果classname是一个字符串
    if classname.find('Linear') != -1:  #判断这个类名中，是否包含"Linear"这个字符串，字符串的find()方法，检索这个字符串中是否包含另一个字符串
        m.weight.data.normal_(0.0, 1.)
        m.bias.data.fill_(0.)


class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.fc1 = nn.Linear(2, 3)   # 隐藏层 3个神经元
        self.fc2 = nn.Linear(3, 4)   # 隐藏层 4个神经元
        self.fc3 = nn.Linear(4, 1)   # 输出层 1个神经元

    def forward(self, x):
        h1 = F.sigmoid(self.fc1(x))  # 之前也尝试过用ReLU作为激活函数, 太容易死亡ReLU了.
        h2 = F.sigmoid(self.fc2(h1))
        h3 = F.sigmoid(self.fc3(h2))
        return h3


net = XOR()
net.apply(weight_init_normal) #相当于net.weight_init_normal()
 #apply方式的调用是递归的，即net这个类和其子类(如果有)，挨个调用一次weight_init_normal()方法。
x = torch.Tensor(x.reshape(-1, 2))
y = torch.Tensor(y.reshape(-1, 1))

# 定义loss function
criterion = nn.BCELoss()  # MSE
# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)  # SGD
# 训练
for epoch in range(500):
    optimizer.zero_grad()   # 清零梯度缓存区
    out = net(x)
    loss = criterion(out, y)
    print(loss)
    loss.backward()
    optimizer.step()  # 更新

# 测试
test = net(x)
print("input is {}".format(x.detach().numpy()))
print('out is {}'.format(test.detach().numpy()))