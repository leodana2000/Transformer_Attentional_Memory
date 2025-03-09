"""
Experiment 2: We know that the scaling is linear in H and want to test linearity in d_head. 
We fix H and d and take d_head varying.
In this setup, we used:
* N=50
* d=10
* H=20 
We find a quadratic relation, meaning that T_0 = C(d)*H*d_head**2.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import reg_lin

data: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_2.csv')
d_head_list = data['d_head'].to_list()
accuracy = data['acc'].to_list()
N = data['N'].to_list()[0]
d = data['d'].to_list()[0]
H = data['para'].to_list()[0]

# Compute the quadratic regression y=ax**2+b.
a, b = reg_lin(np.array(d_head_list)**2, np.array(accuracy))
X_reg = np.linspace(d_head_list[0], d_head_list[-1], 10000)
Y_reg = X_reg**2*a+b

plt.plot(d_head_list, accuracy, label="AoT")
plt.plot(d_head_list, [1/N for _ in range(len(d_head_list))], color='black', label="Random")
plt.plot(d_head_list, [(1-1/N)*(H*d_head+d)/(N**2)+1/N for d_head in d_head_list], label="Our lower bound")
plt.plot(d_head_list, [(1-1/N)*(H*(d_head-1)+1)/(N**2)+1/N for d_head in d_head_list], label="Previous lower bound")
plt.plot(X_reg, Y_reg, color= "C0", linestyle='--', label='Quadratic regression')
plt.xlabel("Head dimension", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.title(f"Scaling law for N={N}, d={d}, H={H}.", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("Images/Exp_2.png", dpi=600)
plt.show()