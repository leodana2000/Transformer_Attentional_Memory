"""
Experiment 5: we want to compare the associative memory scaling laws for attention only and MLP based Transformers. 
We compare our AoT to a Transformer with a single attention head followed by an MLP of width W.
In this setup, we used:
* N=70
* d=10
* H between 1 and 31
* W between 0 and 620
We find that [...].
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


colors = ['C0', 'C1', 'C2']
fig, ax = plt.subplots()
for j, i in enumerate([7, 10, 13]):
    data_att: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_5_{i}_att.csv')
    data_mlp: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_5_{i}_mlp.csv')

    para_list = data_att['para'].to_list()
    accuracy = data_att['acc'].to_list()
    N = data_att['N'].to_list()[0]
    d = data_att['d'].to_list()[0]
    param_count_att = [4*(d**2)*para + d*(2*N+1) for para in para_list]
 
    ax.plot(param_count_att, accuracy, color=colors[j], linestyle='--')

    width_list = data_mlp['width'].to_list()
    accuracy = data_mlp['acc'].to_list()
    param_count_mlp = param_count_att

    ax.plot(param_count_mlp, accuracy, color=colors[j], linestyle='-')

legend_elements = [
    Line2D([0], [0], color='C0', lw=2, label='d=7'),
    Line2D([0], [0], color='C1', lw=2, label='d=10'),
    Line2D([0], [0], color='C2', lw=2, label='d=13'),
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='MLP'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='Attention'),
]

ax.set_xlabel("Parameter count", fontsize=18)
ax.set_ylabel("Accuracy", fontsize=18)
ax.set_title(f"Scaling law for N={N}.", fontsize=18)
ax.legend(handles=legend_elements, fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("Images/Exp_5.png", dpi=400)
plt.show()