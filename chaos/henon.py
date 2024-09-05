import numpy as np
import matplotlib.pyplot as plt

# 参数设置
a = 1.4
b = 0.3
num_points = 10000  # 生成的点的数量

# 初始化 x 和 y
x = np.zeros(num_points)
y = np.zeros(num_points)

# 初始条件
x[0] = 0.1
y[0] = 0.3

# 迭代生成点
for i in range(1, num_points):
    x[i] = 1 - a * x[i - 1]**2 + y[i - 1]
    y[i] = b * x[i - 1]

# 绘制 Hénon 吸引子
plt.figure(figsize=(10, 7))
plt.plot(x, y, ',k', alpha=0.5)
plt.title('Hénon Attractor')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('png/henon.png')
plt.close()
