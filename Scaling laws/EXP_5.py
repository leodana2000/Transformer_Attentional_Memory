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
    data_att: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_1_{i}.csv')
    data_mlp: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_5_{i}.csv')

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
    Line2D([0], [0], color='C0', lw=2, label='d=7'),
    Line2D([0], [0], color='C1', lw=2, label='d=10'),
    Line2D([0], [0], color='C2', lw=2, label='d=13'),
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='MLP'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='Attention'),
]

ax.set_xlabel("Parameter count")
ax.set_ylabel("Accuracy")
ax.set_title(f"Scaling law for N={N}.")
ax.legend(handles=legend_elements)
plt.show()