"""
Experiment 2: since we know that the scaling is linear in H and want to test linearity in d_head, we fix H and d and take d_head varying.
In this setup, we used:
* N=50
* d=10
* H=20 
We find a linear relation, meaning that T_0 = C(d)*H*d_head.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_2.csv')
d_head_list = data['d_head'].to_list()
accuracy = data['acc'].to_list()
N = data['N'].to_list()[0]
d = data['d'].to_list()[0]
H = data['para'].to_list()[0]

# Compute the linear regression y=ax+b.
y = sum(accuracy)/len(accuracy)
x = sum(d_head_list)/len(d_head_list)
Y = np.array(accuracy) - y
X = np.array(d_head_list) - x
a = np.sum(Y*X)/np.sum(X*X)
b = y-a*x
Lin_reg = np.array(d_head_list)*a+b

plt.plot(d_head_list, accuracy, label="AoT")
plt.plot(d_head_list, [1/N for _ in range(len(d_head_list))], color='black', label="Random")
plt.plot(d_head_list, [(1-1/N)*H*d_head/(N**2)+1/N for d_head in d_head_list], label="Our lower bound")
plt.plot(d_head_list, [(1-1/N)*(H*(d_head-1)+1)/(N**2)+1/N for d_head in d_head_list], label="Previous lower bound")
plt.plot(d_head_list, Lin_reg, color= "C0")
plt.xlabel("Head dimension")
plt.ylabel("Accuracy")
plt.title(f"Scaling law for N={N}, d={d}, H={H}.")
plt.legend()
plt.show()

print(f"The slope of the law is {a}.")