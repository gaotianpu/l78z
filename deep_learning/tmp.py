import numpy as np
import matplotlib.pyplot as plt
# import torch 
# import torch.nn as nn
# import torch.nn.functional as F


x = np.linspace(-5, 5, 101) # 生成-5到5之间的51个点的一维元组
print(x)
y = 2 * x * x + 1
plt.plot(x, y)  # 画图
plt.show() 


