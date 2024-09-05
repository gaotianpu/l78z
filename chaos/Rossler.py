import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 定义 Rössler 系统的微分方程
def rossler(t, state, a, b, c):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

# 参数设置
a = 0.2
b = 0.2
c = 5.7

# 初始条件
initial_state = [0.0, 1.0, 0.0]

# 时间跨度
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# 使用 solve_ivp 求解微分方程
solution = solve_ivp(rossler, t_span, initial_state, args=(a, b, c), t_eval=t_eval)

# 获取解
x, y, z = solution.y

# 绘制 Rössler 吸引子
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_title('Rössler Attractor')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.savefig('png/Rossler.png')
plt.close()
