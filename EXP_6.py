"""
Exp 6: We want to compute the scaling laws when d=2 to compare it with Corollary 1. 
In this setup we use:
* N=10
* d=2
* d_head=2
* H ranging from 1 to 21
We find that .
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import reg_lin

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

# Compute the linear regression before saturation.
a, b = reg_lin(linear_H_list, linear_accuracy)
X_reg = np.linspace(linear_H_list[0], linear_H_list[-1], 10000)
Y_reg = X_reg*a+b

# We plot the linear regression
plt.plot(H_list, accuracy, label="AoT")
plt.plot(H_list, [1/N for _ in range(len(H_list))], color='black', label="Random")
plt.plot(H_list, [min((1-1/N)*(H*d_head+d)/(N**2)+1/N,1) for H in H_list], label="Our lower bound")
plt.plot(H_list, [min((1-1/N)*(para*(d_head-1)+1)/(N**2)+1/N,1) for para in H_list], label="Previous lower bound")
plt.plot(X_reg, Y_reg, color= "C0", linestyle='--', label='Linear regression')
plt.xlabel("Number of Heads", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.title(rf"Scaling law for $N=${N}, $d=2$, $d_h=${d_head}.", fontsize=18)
xticks=[1.0, 5.0, 9.0, 13.0, 17.0, 21.0]
xlabels=[str(int(x)) for x in xticks]
plt.xticks(fontsize=14, ticks=xticks,  labels=xlabels)
plt.yticks(fontsize=14)
plt.legend(fontsize=13, loc='upper left')
plt.tight_layout()
plt.savefig("Images/Exp_6.png", dpi=400)
plt.show()