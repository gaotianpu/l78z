import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 定义双涡卷系统的微分方程
def double_scroll(t, state, alpha, beta, gamma):
    x, y, z = state
    dxdt = alpha * (y - x - (x**3)/3)
    dydt = x - y + z
    dzdt = -beta * y - gamma * z
    return [dxdt, dydt, dzdt]

# 参数设置
alpha = 10.0
beta = 14.0
gamma = 0.1

# 初始条件
initial_state = [0.1, 0.0, 0.0]

# 时间跨度
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# 使用 solve_ivp 求解微分方程
solution = solve_ivp(double_scroll, t_span, initial_state, args=(alpha, beta, gamma), t_eval=t_eval)

# 获取解
x, y, z = solution.y

# 绘制双涡卷吸引子
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_title('Double Scroll Attractor')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.savefig('png/double_scroll.png')
plt.close()
