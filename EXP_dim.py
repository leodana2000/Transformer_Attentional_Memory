"""
Exp 1: We want to validate that the scaling law is linear in H, so for different values of d=d_head, we measure the accuracy as H grows. 
In this setup we use:
* N=200
* d=d_head ranging from 40 to 60
We find that before saturation, the accuracy is linear in H.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import reg_lin

for i in [40, 50, 60]:
    data: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_1_{i}_dim.csv')
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

    # We manually compute the linear regression.
    a, b = reg_lin(np.array(linear_H_list), np.array(linear_accuracy))
    X_reg = np.linspace(H_list[0], H_list[-1], 10000)
    Y_reg = np.array(X_reg)*a+b

    # We plot the linear regression
    plt.plot(H_list, accuracy, label="AoT")
    plt.plot(H_list, [1/N for _ in range(len(H_list))], color='black', label="Random")
    plt.plot(H_list, [(1-1/N)*(para*d+d)/(N**2)+1/N for para in H_list], label="Our lower bound")
    plt.plot(H_list, [(1-1/N)*(para*(d-1)+1)/(N**2)+1/N for para in H_list], label="Previous lower bound")
    plt.plot(X_reg, Y_reg, color= "C0", linestyle='--', label='Linear regression')
    plt.xlabel("Number of Heads", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.title(rf"Scaling law for $N=${N}, $d=d_h=${d}.", fontsize=18)
    xticks=[1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    xlabels=[str(int(x)) for x in xticks]
    plt.xticks(fontsize=14, ticks=xticks,  labels=xlabels)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper left', fontsize=14)
    plt.tight_layout()
    if d==50:
        plt.savefig("Images/Exp_1_dim.png", dpi=400)
    plt.show()

"""
Experiment 2: since we know that the scaling is linear in H and want to test linearity in d_head, we fix H and d and take d_head varying.
In this setup, we used:
* N=200
* d=50
* H=20 
We find a linear relation, meaning that T_0 = C(d)*H*d_head.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(f'Scaling laws/Data_exp_2_dim.csv')
d_head_list = data['d_head'].to_list()
accuracy = data['acc'].to_list()
N = data['N'].to_list()[0]
d = data['d'].to_list()[0]
H = data['para'].to_list()[0]


# Compute the linear regression y=ax+b.
a, b = reg_lin(np.array(d_head_list), np.array(accuracy))
X_reg_1 = np.linspace(d_head_list[0], d_head_list[-1], 10000)
Y_reg_1 = X_reg_1*a+b

# Compute the quadratic regression y=ax**2+b.
a, b = reg_lin(np.array(d_head_list)**2, np.array(accuracy))
X_reg_2 = np.linspace(d_head_list[0], d_head_list[-1], 10000)
Y_reg_2 = X_reg_2**2*a+b

plt.plot(d_head_list, accuracy, label="AoT")
plt.plot(d_head_list, [1/N for _ in range(len(d_head_list))], color='black', label="Random")
plt.plot(d_head_list, [(1-1/N)*(H*d_head+d)/(N**2)+1/N for d_head in d_head_list], label="Our lower bound")
plt.plot(d_head_list, [(1-1/N)*(H*(d_head-1)+1)/(N**2)+1/N for d_head in d_head_list], label="Previous lower bound")
#plt.plot(X_reg_1, Y_reg_1, color= "C0", linestyle='dashed', label='Linear regression')
plt.plot(X_reg_2, Y_reg_2, color= "C0", linestyle='dashed', label='Quadratic regression')
plt.xlabel("Head dimension", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.title(f"Scaling law for N={N}, d={d}, H={H}.", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper left', fontsize=14)
plt.tight_layout()
plt.savefig("Images/Exp_2_dim.png", dpi=400)
plt.show()

"""
Experiment 5: we want to compare the associative memory scaling laws for attention only and MLP based Transformers. 
We compare our AoT to a Transformer with a single attention head followed by an MLP of width W.
In this setup, we used:
* N=200
* d between 40 and 60
* H between 1 and 31
* W between 0 and 620
We find that [...].
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


colors = ['C0', 'C1', 'C2'] 
line_styles = ['--', '-'] 
fig, ax = plt.subplots()
for j, i in enumerate([40, 50, 60]):
    data_att: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_1_{i}_dim.csv')
    data_mlp: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_5_{i}_dim.csv')

    para_list = data_att['para'].to_list()[:6]
    accuracy = data_att['acc'].to_list()[:6]
    N = data_att['N'].to_list()[0]
    d = data_att['d'].to_list()[0]
    param_count_att = [4*(d**2)*para+2*d*(N+1) for para in para_list]

    ax.plot(param_count_att, accuracy, color=colors[j], linestyle='--')

    width_list = data_mlp['width'].to_list()
    accuracy = data_mlp['acc'].to_list()
    N = data_mlp['N'].to_list()[0]
    d = data_mlp['d'].to_list()[0]
    param_count_mlp = [4*(d**2)+2*d*width+2*d*(N+1) for width in width_list]

    ax.plot(param_count_mlp, accuracy, color=colors[j], linestyle='-')

legend_elements = [
    Line2D([0], [0], color='C0', lw=2, label='d=40'),
    Line2D([0], [0], color='C1', lw=2, label='d=50'),
    Line2D([0], [0], color='C2', lw=2, label='d=60'),
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='MLP'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='Attention'),
]

ax.set_xlabel("Parameter count", fontsize=18)
ax.set_ylabel("Accuracy", fontsize=18)
ax.set_title(f"Scaling law for N={N}.", fontsize=18)
ax.legend(handles=legend_elements, fontsize=14)
plt.xticks(fontsize=11.5)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("Images/Exp_5_dim.png", dpi=400)
plt.show()