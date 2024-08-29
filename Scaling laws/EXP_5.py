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

for i in [2, 7, 10]:
    data_att: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_1_{i}.csv')
    data_mlp: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_5_{i}.csv')

    para_list = data_att['para'].to_list()[:6]
    accuracy = data_att['acc'].to_list()[:6]
    N = data_att['N'].to_list()[0]
    d = data_att['d'].to_list()[0]
    param_count_att = [4*(d**2)*para+2*d*(N+1) for para in para_list]

    plt.plot(param_count_att, accuracy, label="Attention")

    width_list = data_mlp['width'].to_list()
    accuracy = data_mlp['acc'].to_list()
    N = data_mlp['N'].to_list()[0]
    d = data_mlp['d'].to_list()[0]
    param_count_mlp = [4*(d**2)+2*d*width+2*d*(N+1) for width in width_list]

    plt.plot(param_count_mlp, accuracy, label="MLP")
    plt.xlabel("Parameter count")
    plt.ylabel("Accuracy")
    plt.title(f"Scaling law for N={N}, d={d}.")
    plt.legend()
    plt.show()