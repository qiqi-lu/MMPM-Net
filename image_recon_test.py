import numpy as np
import matplotlib.pyplot as plt

def _weights(z, y, x):
    W = np.zeros([z, y, x])
    for k in range(z):
        for i in range(y):
            for j in range(x):
                d = ((i) / y - 0.5)**2 + ((j) / x - 0.5)**2 + ((k) / z - 0.5)**2
                W[k, i, j] = 1 / (1 + 220 * d)**16
    return W

z,y,x=128,128,128
w = _weights(z,y,x)
plt.figure()
plt.imshow(w[64],cmap='jet'),plt.colorbar(fraction=0.022)
plt.savefig('figures/tmp')