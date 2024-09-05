import numpy as np
import matplotlib.pyplot as plt

# 定义参数
r = 3.7  # 增长率参数

def run(r):
    x0 = 0.5  # 初始种群比例
    iterations = 100  # 迭代次数

    # 创建数组来存储每一代的种群比例
    x_values = np.zeros(iterations)
    x_values[0] = x0

    # 迭代更新种群比例
    for i in range(1, iterations):
        x_values[i] = r * x_values[i - 1] * (1 - x_values[i - 1])

    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, marker='o', linestyle='-', color='b')
    plt.title(f"Logistic Map (r={r})")
    plt.xlabel("Generation")
    plt.ylabel("Population Proportion (x)")
    plt.grid(True)
    plt.savefig(f'png/Robert_May.{r}.png')
    plt.close()

for r in [1,2,2.5,3,3.1,3.5,3.7,4]:
    run(r)
