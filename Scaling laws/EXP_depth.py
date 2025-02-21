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

plot_all = True

Scaling_coef = []
for i in [4, 7, 10]:
    data: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_1_{i}_depth.csv')
    H_list = data['para'].to_list()
    accuracy = data['acc'].to_list()
    N = data['N'].to_list()[0]
    d = data['d'].to_list()[0]

    # We filter out if the last points are constant at accuracy 1 to compute the linear scaling before it saturates.
    for i, acc in enumerate(accuracy):
        if (acc > 0.95) or (i == len(accuracy)-1):
            linear_accuracy = accuracy[:i+1]
            H_list = H_list[:i+1]
            break

    # We manually compute the linear regression.
    y = sum(linear_accuracy)/len(linear_accuracy)
    x = sum(H_list)/len(H_list)
    Y = np.array(linear_accuracy) - y
    X = np.array(H_list) - x
    a = np.sum(Y*X)/np.sum(X*X)
    Scaling_coef.append(a*(N**2))
    b = y-a*x
    Z = np.array(H_list)*a+b

    # We plot the linear regression
    if plot_all:
        plt.plot(H_list, linear_accuracy, label="AoT")
        plt.plot(H_list, [1/N for _ in range(len(H_list))], color='black', label="Random")
        plt.plot(H_list, [(1-1/N)*(para*d+d)/(N**2)+1/N for para in H_list], label="Our lower bound")
        plt.plot(H_list, [(1-1/N)*(para*(d-1)+1)/(N**2)+1/N for para in H_list], label="Previous lower bound")
        plt.plot(H_list, Z, color= "C0", linestyle='--', label='Linear regression')
        plt.xlabel("Number of Heads per layers")
        plt.ylabel("Accuracy")
        plt.title(f"Scaling law for N={N}, d=d_head={d}, {5} layers.")
        plt.legend()
        plt.show()

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

data = pd.read_csv(f'Scaling laws/Data_exp_2_depth.csv')
d_head_list = data['d_head'].to_list()
accuracy = data['acc'].to_list()
N = data['N'].to_list()[0]
d = data['d'].to_list()[0]
H = data['para'].to_list()[0]

# Compute the linear regression y=ax+b.
y = sum(np.array(accuracy))/len(accuracy)
x = sum(np.array(d_head_list)**2)/len(d_head_list)
Y = np.array(accuracy) - y
X = np.array(d_head_list)**2 - x
a = np.sum(Y*X)/np.sum(X*X)
b = y-a*x
Quad_reg = np.array(d_head_list)**2*a+b

plt.plot(d_head_list, accuracy, label="AoT")
plt.plot(d_head_list, [1/N for _ in range(len(d_head_list))], color='black', label="Random")
plt.plot(d_head_list, [(1-1/N)*(H*d_head+d)/(N**2)+1/N for d_head in d_head_list], label="Our lower bound")
plt.plot(d_head_list, [(1-1/N)*(H*(d_head-1)+1)/(N**2)+1/N for d_head in d_head_list], label="Previous lower bound")
plt.plot(d_head_list, Quad_reg, color= "C0", linestyle='--', label='Quadratic regression')
plt.xlabel("Head dimension")
plt.ylabel("Accuracy")
plt.title(f"Scaling law for N={N}, d={d}, H={H}, 5 layers.")
plt.legend()
plt.show()

print(f"The slope of the law is {a}.")

"""
Experiment 5: we want to compare the associative memory scaling laws for attention only and MLP based Transformers. 
We compare our AoT to a Transformer with a single attention head followed by an MLP of width W.
In this setup, we used:
* N=50
* d=10
* H between 1 and 26
* W between 2*10*1 and 2*10*26
We find that both scaling are linear and of the same size effect.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


colors = ['C0', 'C1', 'C2'] 
line_styles = ['--', '-'] 
fig, ax = plt.subplots()
for j, i in enumerate([4, 7, 10]):
    data_att: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_1_{i}_depth.csv')
    data_mlp: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_5_{i}_depth.csv')

    para_list = data_att['para'].to_list()[:6]
    accuracy = data_att['acc'].to_list()[:6]
    N = data_att['N'].to_list()[0]
    d = data_att['d'].to_list()[0]
    param_count_att = [4*(d**2)*para*5+2*d*(N+1) for para in para_list]

    ax.plot(param_count_att, accuracy, color=colors[j], linestyle='--')

    width_list = data_mlp['width'].to_list()
    accuracy = data_mlp['acc'].to_list()
    N = data_mlp['N'].to_list()[0]
    d = data_mlp['d'].to_list()[0]
    param_count_mlp = [4*(d**2)*5+2*d*width*5+2*d*(N+1) for width in width_list]

    ax.plot(param_count_mlp, accuracy, color=colors[j], linestyle='-')

legend_elements = [
    Line2D([0], [0], color='C0', lw=2, label='d=7'),
    Line2D([0], [0], color='C1', lw=2, label='d=10'),
    Line2D([0], [0], color='C2', lw=2, label='d=13'),
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='MLP'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='Attention'),
]

ax.set_xlabel("Parameter count")
ax.set_ylabel("Accuracy")
ax.set_title(f"Scaling law for N={N} and 5 layers.")
ax.legend(handles=legend_elements)
plt.show()