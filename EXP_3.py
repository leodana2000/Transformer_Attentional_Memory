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
from utils import reg_lin

data: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_3.csv')
d_list = data['d'].to_list()
accuracy = data['acc'].to_list()
N = data['N'].to_list()[0]
d_head = data['d_head'].to_list()[0]
H = data['para'].to_list()[0]

# Compute the linear regression y=ax+b for the first half.
a, b = reg_lin(np.array(d_list[:6]), np.array(accuracy[:6]))
X_reg_1 = np.linspace(d_list[0], d_list[5], 10000)
Y_reg_1 = X_reg_1*a+b

# Compute the linear regression y=ax+b for the second half.
a, b = reg_lin(np.array(d_list[5:]), np.array(accuracy[5:]))
X_reg_2 = np.linspace(d_list[5], d_list[-1], 10000)
Y_reg_2 = X_reg_2*a+b

plt.plot(d_list, accuracy, label="AoT")
plt.plot(d_list, [1/N for _ in range(len(d_list))], color='black', label="Random")
plt.plot(d_list, [(1-1/N)*(H*d_head+d)/(N**2)+1/N for d in d_list], label="Our lower bound")
plt.plot(d_list, [(1-1/N)*(H*(d_head-1)+1)/(N**2)+1/N for d in d_list], label="Previous lower bound")
plt.plot(X_reg_1, Y_reg_1, color="C0", linestyle='--', label='Linear regressions')
plt.plot(X_reg_2, Y_reg_2, color="C0", linestyle='--')
plt.xlabel("Embedding dimension", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.title(rf"Scaling law for $N=${N}, $d_h=${d_head}, $H=${H}.", fontsize=18)
xticks=[5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
xlabels=[str(int(x)) for x in xticks]
plt.xticks(fontsize=14, ticks=xticks,  labels=xlabels)
plt.yticks(fontsize=14)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig("Images/Exp_3.png", dpi=600)
plt.show()