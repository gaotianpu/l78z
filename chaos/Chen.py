import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def chen(t, state, a, b, c):
    x, y, z = state
    dxdt = a * (y - x)
    dydt = (c - a) * x - x * z + c * y
    dzdt = x * y - b * z
    return [dxdt, dydt, dzdt]

a, b, c = 35, 3, 28
initial_state = [0.1, 0.0, 0.0]
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

solution = solve_ivp(chen, t_span, initial_state, args=(a, b, c), t_eval=t_eval)
x, y, z = solution.y

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_title('Chen Attractor')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.savefig('png/Chen.png')
plt.close()

