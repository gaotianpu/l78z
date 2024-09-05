import numpy as np
import matplotlib.pyplot as plt

# 定义多项式 f(z) = z^3 - 1 及其导数 f'(z)
def f(z):
    return z**3 - 1

def df(z):
    return 3 * z**2

# 定义牛顿迭代
def newton_method(z, max_iter=50, tol=1e-6):
    for i in range(max_iter):
        dz = f(z) / df(z)
        z = z - dz
        # 如果迭代的值足够接近一个根，提前停止
        if np.abs(dz) < tol:
            break
    return z

# 定义绘图区域 (复平面)
re_min, re_max = -2, 2
im_min, im_max = -2, 2
resolution = 500

# 生成复平面的网格点
real = np.linspace(re_min, re_max, resolution)
imag = np.linspace(im_min, im_max, resolution)
X, Y = np.meshgrid(real, imag)
Z = X + 1j * Y  # 将实数和虚数组合成复数

# 定义三个根
roots = [1, np.exp(2j * np.pi / 3), np.exp(-2j * np.pi / 3)]

# 初始化颜色矩阵
root_indices = np.zeros(Z.shape, dtype=int)

# 迭代牛顿法并标记每个点收敛到哪个根
for i in range(resolution):
    for j in range(resolution):
        z = Z[i, j]
        result = newton_method(z)
        # 根据结果判断 z 收敛到哪个根
        distances = [np.abs(result - root) for root in roots]
        root_indices[i, j] = np.argmin(distances)

# 定义颜色映射
colors = ['red', 'green', 'blue']

# 创建一个彩色图像，显示不同区域收敛到不同的根
plt.figure(figsize=(8, 8))
plt.imshow(root_indices, extent=(re_min, re_max, im_min, im_max), cmap=plt.get_cmap('brg'))
plt.title("Newton Fractal for $z^3 - 1$")
plt.xlabel("Re")
plt.ylabel("Im")
plt.savefig('png/Hubbard.png')
plt.close()
