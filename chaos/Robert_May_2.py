import numpy as np
import matplotlib.pyplot as plt

# 定义参数
r_values = np.linspace(2.5, 4.0, 10000)  # r参数的范围
iterations = 1000  # 总迭代次数
last = 100  # 取后几次的值来绘图
x = 1e-5 * np.ones(len(r_values))  # 初始种群比例

# 数组存储 r 和相应的 x 值
r_list = []
x_list = []

# 迭代更新种群比例
for i in range(iterations):
    x = r_values * x * (1 - x)
    if i >= (iterations - last):
        r_list.extend(r_values)
        x_list.extend(x)

# 绘制树形分叉图
plt.figure(figsize=(12, 8))
plt.plot(r_list, x_list, ',k', alpha=0.25)
plt.xlim(2.5, 4)
plt.title("Bifurcation Diagram")
plt.xlabel("Growth Rate (r)")
plt.ylabel("Population Proportion (x)")
plt.grid(True)
# plt.show()
plt.savefig('png/Robert_May_2.tree.png')
plt.close()