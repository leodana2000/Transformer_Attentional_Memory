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
half_accuracy = accuracy[:6]
half_d_list = d_list[:6]
y = np.mean(half_accuracy)
x = np.mean(half_d_list)
Y = np.array(half_accuracy) - y
X = np.array(half_d_list) - x
a = np.sum(Y*X)/np.sum(X*X)
b = y-a*x
Half_Lin_reg = np.array(half_d_list)*a+b
print(f"Linear regression. a: {a}, b: {b}.")

# Compute the linear regression y=ax+b for the second half.
accuracy_half = accuracy[5:]
d_list_half = d_list[5:]
y = np.mean(accuracy_half)
x = np.mean(d_list_half)
Y = np.array(accuracy_half) - y
X = np.array(d_list_half) - x
a = np.sum(Y*X)/np.sum(X*X)
b = y-a*x
Lin_reg_Half = np.array(d_list_half)*a+b
print(f"Linear regression. a: {a}, b: {b}.")

plt.plot(d_list, accuracy, label="AoT")
plt.plot(d_list, [1/N for _ in range(len(d_list))], color='black', label="Random")
plt.plot(d_list, [(1-1/N)*H*d_head/(N**2)+1/N for d in d_list], label="Our lower bound")
plt.plot(d_list, [(1-1/N)*(H*(d_head-1)+1)/(N**2)+1/N for d in d_list], label="Previous lower bound")
plt.plot(half_d_list, Half_Lin_reg, color="C0")
plt.plot(d_list_half, Lin_reg_Half, color="C0")
plt.xlabel("Embedding dimension")
plt.ylabel("Accuracy")
plt.title(f"Scaling law for N={N}, d_head={d_head}, H={H}.")
plt.legend()
plt.show()