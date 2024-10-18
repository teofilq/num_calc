import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


data = np.genfromtxt('L1/regresie.csv', delimiter=",")
z = data[:, 0] 
w = data[:, 1]  
A = np.vstack([z, np.ones(len(z))]).T
print(A)
b = w
sol = linalg.lstsq(A, b)
alpha, beta = sol[0]
print(b)
print(sol)
plt.scatter(z, w) 
plt.plot(z, alpha * z + beta) 
plt.xlabel("z")
plt.ylabel("w")
plt.title(f"w = {alpha} * z + {beta}")
plt.show()
