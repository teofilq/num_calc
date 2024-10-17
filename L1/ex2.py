import numpy as np

def back_substitution(U, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    return x


n = 6
A = np.random.randn(n,n)
b = np.random.randn(n)

A_orig = A.copy()
b_orig = b.copy()

for k in range(0, n-1):
    for i in range(k+1, n):
        factor = -A[i][k] / A[k][k]
        A[i][k] = 0 
        b[i] = b[i] + b[k] * factor
        for j in range(k+1, n):
            A[i][j] = A[i][j] + A[k][j] * factor

U = np.triu(A)

x = back_substitution(U, b)


print("Soluția:", x)

x_check = np.linalg.solve(A_orig, b_orig)
print("Solutia este", np.allclose(x ,x_check))
