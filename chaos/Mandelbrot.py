import numpy as np
import matplotlib.pyplot as plt

# 绘制曼德博集合的代码
def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n

# 定义图像的尺寸和迭代次数
width, height = 800, 800
max_iter = 100

# 定义显示区域的范围
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5

# 创建图像
image = np.zeros((height, width))

# 生成图像
for x in range(width):
    for y in range(height):
        real = xmin + (x / width) * (xmax - xmin)
        imag = ymin + (y / height) * (ymax - ymin)
        c = complex(real, imag)
        color = mandelbrot(c, max_iter)
        image[y, x] = color

# 显示曼德博集合
plt.figure(figsize=(10, 10))
plt.imshow(image, extent=[xmin, xmax, ymin, ymax], cmap='twilight_shifted')
plt.colorbar()
plt.title("Mandelbrot Set")
plt.savefig('png/Mandelbrot.png')
plt.close()
