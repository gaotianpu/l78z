
import numpy as np
import matplotlib.pyplot as plt

def ikeda(n, u=0.918):
    x, y = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        t = 0.4 - 6 / (1 + x[i-1]**2 + y[i-1]**2)
        x[i] = 1 + u * (x[i-1] * np.cos(t) - y[i-1] * np.sin(t))
        y[i] = u * (x[i-1] * np.sin(t) + y[i-1] * np.cos(t))
    return x, y

n = 10000
x, y = ikeda(n)

plt.figure(figsize=(10, 7))
plt.plot(x, y, ',k', alpha=1)
plt.title('Ikeda Attractor')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.savefig('png/Ikeda.png')
plt.close()



 