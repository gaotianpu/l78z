import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def chua(t, state, alpha, beta, m0, m1):
    x, y, z = state
    f_x = m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))
    dxdt = alpha * (y - x - f_x)
    dydt = x - y + z
    dzdt = -beta * y
    return [dxdt, dydt, dzdt]

alpha, beta, m0, m1 = 9.0, 14.286, -1.143, -0.714
initial_state = [0.7, 0.0, 0.0]
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

solution = solve_ivp(chua, t_span, initial_state, args=(alpha, beta, m0, m1), t_eval=t_eval)
x, y, z = solution.y

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_title("Chua's Attractor")
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.savefig('png/Chua.png')
plt.close()
