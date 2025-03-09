"""
Exp 1: We want to validate that the scaling law is linear in H, so for different values of d=d_head, we measure the accuracy as H grows. 
In this setup we use:
* N=50
* d=d_head ranging from 3 to 13
We find that before saturation, the accuracy is linear in H.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import reg_lin

Scaling_coef = []
for i in range(3, 13+1):
    data: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_1_{i}.csv')
    H_list = data['para'].to_list()
    accuracy = data['acc'].to_list()
    N = data['N'].to_list()[0]
    d = data['d'].to_list()[0]

    # We filter out if the last points are constant at accuracy 1 to compute the linear scaling before it saturates.
    for i, acc in enumerate(accuracy):
        if (acc > 0.95) or (i == len(accuracy)-1):
            linear_accuracy = accuracy[:i+1]
            linear_H_list = H_list[:i+1]
            break

    # We compute the linear regression.
    a, b = reg_lin(np.array(linear_H_list), np.array(linear_accuracy))
    continuous_X = np.linspace(H_list[0], H_list[-1], 10000)
    continuous_Y = np.minimum(1, continuous_X*a+b)

    Scaling_coef.append(a)

    # We plot the linear regression
    plt.plot(H_list, accuracy, label="AoT")
    plt.plot(H_list, [1/N for _ in range(len(H_list))], color='black', label="Random")
    plt.plot(H_list, [(1-1/N)*(para*d+d)/(N**2)+1/N for para in H_list], label="Our lower bound")
    plt.plot(H_list, [(1-1/N)*(para*(d-1)+1)/(N**2)+1/N for para in H_list], label="Previous lower bound")
    plt.plot(continuous_X, continuous_Y, color= "C0", linestyle='--', label='Linear regression')
    plt.xlabel("Number of Heads", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.title(rf"Scaling law for $N=${N}, $d=d_h=${d}.", fontsize=18)
    xticks=[i for i in range(1, 31+1, 5)]
    xlabels=[str(int(x)) for x in xticks]
    plt.xticks(fontsize=14, ticks=xticks,  labels=xlabels)
    plt.yticks(fontsize=14)
    if d<10:
        plt.legend(fontsize=13, loc="upper left")
    else:
        plt.legend(fontsize=13, loc="center right")
    plt.tight_layout()
    if d==10:
        plt.savefig("Images/Exp_1.png", dpi=400)
    plt.show()


"""
Exp 4: We validate experiment 2 and 3 by measuring the scaling laws with d=d_head. Using the scaling laws from Exp_1, 
we compute the linear scaling before saturation, and use it for a scaling laws of its own.
In this setup we use:
* N=50
* d=d_head
We find a cubic scaling for the coefficient, meaning that the accuracy has the scaling C*H*d**3.
"""

d_list = np.array([d for d in range(3, 13+1)])
d_continuous = np.linspace(3, 13, 10000)

plt.plot(d_list, Scaling_coef, label="Empirical coefficient")

a, b = reg_lin(d_list**2, Scaling_coef)
plt.plot(d_continuous, a*d_continuous**2+b, label="Quadratic approx", linestyle="dashed")
print(f"Quadratic regression. a:{a}, b:{b}")

a, b = reg_lin(d_list**3, Scaling_coef)
plt.plot(d_continuous, a*d_continuous**3+b, label="Cubic approx", linestyle="dashed")
print(f"Cubic regression. a:{a}, b:{b}")

plt.xlabel("Embedding dimension", fontsize=18)
plt.ylabel("Scaling law coefficient", fontsize=18)
plt.title(f"Scaling laws in d", fontsize=18)
xticks=[3, 5, 7, 9, 11, 13]
xlabels=[str(int(x)) for x in xticks]
plt.xticks(fontsize=14, ticks=xticks,  labels=xlabels)
plt.yticks(fontsize=14)
plt.legend(fontsize=15, loc="upper left")
plt.tight_layout()
plt.savefig("Images/Exp_4.png", dpi=400)
plt.show()