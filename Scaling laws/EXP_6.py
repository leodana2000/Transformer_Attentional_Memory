"""
Exp 6: We want to compute the scaling laws when d=2 to compare it with Corollary 1. 
In this setup we use:
* N=10
* d=2
* d_head=5 
* H ranging from 1 to 21
We find that .
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Scaling_coef = []
data: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_6.csv')
H_list = data['para'].to_list()
accuracy = data['acc'].to_list()
N = data['N'].to_list()[0]
d = data['d'].to_list()[0]
d_head = data['d_head'].to_list()[0]

# We filter out if the last points are constant at accuracy 1 to compute the linear scaling before it saturates.
for i, acc in enumerate(accuracy):
    if (acc > 0.95) or (i == len(accuracy)-1):
        linear_accuracy = accuracy[:i+1]
        linear_H_list = H_list[:i+1]
        break

# We manually compute the linear regression.
y = sum(linear_accuracy)/len(linear_accuracy)
x = sum(linear_H_list)/len(linear_H_list)
Y = np.array(linear_accuracy) - y
X = np.array(linear_H_list) - x
a = np.sum(Y*X)/np.sum(X*X)
Scaling_coef.append(a*(N**2))
b = y-a*x
Z = np.array(linear_H_list)*a+b
print(a/(d_head/(N**2)*(1-1/N)))

# We plot the linear regression
plt.plot(H_list, accuracy, label="AoT")
plt.plot(H_list, [1/N for _ in range(len(H_list))], color='black', label="Random")
plt.plot(H_list, [min((1-1/N)*(H*d_head+d)/(N**2)+1/N,1) for H in H_list], label="Our lower bound")
plt.plot(H_list, [min((1-1/N)*(para*(d_head-1)+1)/(N**2)+1/N,1) for para in H_list], label="Previous lower bound")
plt.plot(linear_H_list, Z, color= "C0", linestyle='--', label='Linear regression')
plt.xlabel("Number of Heads")
plt.ylabel("Accuracy")
plt.title(f"Scaling law for N={N}, d=2, d_head={d_head}.")
plt.legend()
plt.show()