import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def rabinovich_fabrikant(t, state, alpha, gamma):
    x, y, z = state
    dxdt = y * (z - 1 + x**2) + gamma * x
    dydt = x * (3 * z + 1 - x**2) + gamma * y
    dzdt = -2 * z * (alpha + x * y)
    return [dxdt, dydt, dzdt]

# 参数设置
alpha, gamma = 0.98, 0.1

# 初始条件
initial_state = [0.1, 0.0, 0.0]

# 时间跨度
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# 使用 solve_ivp 求解微分方程
solution = solve_ivp(rabinovich_fabrikant, t_span, initial_state, args=(alpha, gamma), t_eval=t_eval)

# 获取解
x, y, z = solution.y

# 绘制 Rabinovich-Fabrikant 吸引子
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_title('Rabinovich-Fabrikant Attractor')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.savefig('png/rabinovich_fabrikant.png')
plt.close()



