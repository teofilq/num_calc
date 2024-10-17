from decimal import Decimal

def seq(x0, k_max):
    x_vals = [Decimal(x0)]
    for k in range(k_max):
        xk = x_vals[-1]
        if Decimal('0') <= xk <= Decimal('0.5'):
            x_next = 2 * xk
        else:
            x_next = 2 * xk - 1
        x_vals.append(x_next)
    return x_vals

x0 = Decimal('0.1')
k = 60
seq_vals = seq(x0, k)
print(seq_vals[-1])
print()
print(seq_vals)

