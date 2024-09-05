import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def duffing(t, state, delta, alpha, beta, gamma, omega):
    x, y = state
    dxdt = y
    dydt = x - alpha * x**3 - delta * y + gamma * np.cos(omega * t)
    return [dxdt, dydt]

delta, alpha, beta, gamma, omega = 0.2, 1.0, 1.0, 0.3, 1.2
initial_state = [1.0, 0.0]
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

solution = solve_ivp(duffing, t_span, initial_state, args=(delta, alpha, beta, gamma, omega), t_eval=t_eval)
x, y = solution.y

plt.figure(figsize=(10, 7))
plt.plot(x, y, lw=0.5)
plt.title('Duffing Attractor')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()
plt.savefig('png/Duffing.png')
plt.close()

 