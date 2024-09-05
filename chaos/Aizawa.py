import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def aizawa(t, state, a, b, c, d, e, f):
    x, y, z = state
    dxdt = (z - b) * x - d * y
    dydt = d * x + (z - b) * y
    dzdt = c + a * z - (z ** 3) / 3 - (x ** 2 + y ** 2) * (1 + e * z) + f * z * (x ** 3)
    return [dxdt, dydt, dzdt]

a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
initial_state = [0.1, 0.0, 0.0]
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

solution = solve_ivp(aizawa, t_span, initial_state, args=(a, b, c, d, e, f), t_eval=t_eval)
x, y, z = solution.y

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.5)
ax.set_title('Aizawa Attractor')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.savefig('png/Aizawa.png')
plt.close()
