"""
Experiment 3: since we know that the scaling is linear in H and want to find the scaling in d, we fix H and d_head and take d varying.
In this setup, we used:
* N=50
* d_head=10
* H=20 
We find a linear relation, meaning that T_0 = C(d)*H*d_head/(N**2).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_3.csv')
d_list = data['d'].to_list()
accuracy = data['acc'].to_list()
N = data['N'].to_list()[0]
d_head = data['d_head'].to_list()[0]
H = data['para'].to_list()[0]

# Compute the linear regression y=ax+b for the first half.
y = np.mean(accuracy[:6])
x = np.mean(d_list[:6])
Y = np.array(accuracy[:6]) - y
X = np.array(d_list[:6]) - x
a = np.sum(Y*X)/np.sum(X*X)
b = y-a*x
Half_Lin_reg = np.array(d_list[:6])*a+b
print(f"Linear regression. a: {a}, b: {b}.")

# Compute the linear regression y=ax+b for the first half.
y = np.mean(accuracy[5:])
x = np.mean(np.array(d_list[5:]))
Y = np.array(accuracy[5:]) - y
X = np.array(d_list[5:]) - x
a = np.sum(Y*X)/np.sum(X*X)
b = y-a*x
Lin_Half_reg = np.array(d_list[5:])*a+b
print(f"Linear regression. a: {a}, b: {b}.")

plt.plot(d_list, accuracy, label="AoT")
plt.plot(d_list, [1/N for _ in range(len(d_list))], color='black', label="Random")
plt.plot(d_list, [(1-1/N)*(H*d_head+d)/(N**2)+1/N for d in d_list], label="Our lower bound")
plt.plot(d_list, [(1-1/N)*(H*(d_head-1)+1)/(N**2)+1/N for d in d_list], label="Previous lower bound")
plt.plot(d_list[:6], Half_Lin_reg, color="C0", linestyle='--', label='Linear regressions')
plt.plot(d_list[5:], Lin_Half_reg, color="C0", linestyle='--')
plt.xlabel("Embedding dimension")
plt.ylabel("Accuracy")
plt.title(f"Scaling law for N={N}, d_head={d_head}, H={H}.")
plt.legend()
plt.show()