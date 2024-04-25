import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List

"""
The default for the plots are N=50, d=5, h=100, nb_head=1, nb_layers=1, para=1.
"""

"""# print the average d over N
data_d_N: pd.DataFrame = pd.read_csv('scaling_d_N_new.csv')
for N in [10, 20, 30, 50, 100]:
    group = data_d_N[data_d_N['N'] == N].groupby(['d'])
    loss_group = group.mean()['loss']
    X = [d for d in loss_group.index.to_list()]
    Z = loss_group.to_list()
    #minZ = min(Z)
    #Z = [z-minZ for z in Z]
    plt.plot(X, Z, label=f'N={N}')
    plt.xlabel('d')
    plt.ylabel('KL divergence') #(rescaled)')
plt.title('Scaling law on the residual dimension (d) and the number of tokens (N)')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.show()"""


"""# print the average h over d
data_d_h: pd.DataFrame = pd.read_csv('scaling_d_h_new.csv')
for d in [2, 4, 6, 8, 10]:
    group = data_d_h[data_d_h['d'] == d].groupby(['h'])
    loss_group = group.mean()['loss']
    X = [h for h in loss_group.index.to_list()]
    Z = loss_group.to_list()
    plt.plot(X, Z, label=f'd={d}')
plt.xlabel('Ratio h/d')
plt.ylabel('KL divergence')
plt.title('Scaling law on the residual dimension (d) and the hidden dimension (h)')
plt.legend()
plt.show()"""


"""# print the average nb_layers over d
data_d_layer: pd.DataFrame = pd.read_csv('scaling_d_layer_new.csv')
list_d: List[int] = data_d_layer.groupby(['d']).mean()['loss'].index.to_list()
for d in list_d:
    group = data_d_layer[data_d_layer['d'] == d].groupby(['nb_layers'])
    loss_group = group.mean()['loss']
    X = loss_group.index.to_list()
    Z = loss_group.to_list()
    plt.plot(X, Z, label=f'd={d}')
plt.xlabel('Number of layers')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel('KL divergence')
plt.title('Scaling law on the residual dimension (d) and the number of layers')
plt.legend()
plt.show()"""


"""# print the average head over d
data_d_head: pd.DataFrame = pd.read_csv('scaling_d_head_new.csv')
list_d = data_d_head.groupby(['d']).mean()['loss'].index.to_list()
for d in list_d:
    group = data_d_head[data_d_head['d'] == d].groupby(['nb_head'])
    loss_group = group.mean()['loss']
    X = loss_group.index.to_list()
    Z = loss_group.to_list()
    plt.plot(X, Z, label=f'd={d}')
plt.xlabel('Number of heads per self-attention')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel('KL divergence')
plt.title('Scaling law on the residual dimension (d) \n and the number of heads, with 100 tokens')
plt.legend()
plt.show()"""


"""# print the average hidden dimension over the number of parallel heads
data_h_para: pd.DataFrame = pd.read_csv('scaling_h_para_new.csv')
list_h: List[int] = data_h_para.groupby(['h']).mean()['loss'].index.to_list()
for h in list_h:
    group = data_h_para[data_h_para['h'] == h].groupby(['para_head'])
    loss_group = group.mean()['loss']
    X = loss_group.index.to_list()
    Z = loss_group.to_list()
    plt.plot(X, Z, label=f'h={h}')
plt.xlabel('Number of parallel self-attention')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel('KL divergence')
plt.title('Scaling law on the hidden dimension (h) \n and the number of parallel self-attention')
plt.legend()
plt.show()"""

"""# print the effect of the entropy
data_lamb: pd.DataFrame = pd.read_csv('scaling_lamb_new.csv')
group = data_lamb.groupby(['lamb'])
loss_group = group.mean()['loss']
X = loss_group.index.to_list()
Z = loss_group.to_list()
plt.plot(X, Z)
plt.xlabel('Entropy of the distribution')
plt.ylabel('KL divergence')
plt.title('Scaling law on the entropy of the distribution')
plt.show()"""

"""# print the average h over layers
data_h_layer: pd.DataFrame = pd.read_csv('scaling_h_layer_new.csv')
for nb_layers in [1, 2, 4, 5]:
    group = data_h_layer[data_h_layer['nb_layers'] == nb_layers].groupby(['h'])
    loss_group = group.mean()['loss']
    X = [h for h in loss_group.index.to_list()]
    Z = loss_group.to_list()
    plt.plot(X, Z, label=f'l={nb_layers}')
plt.xlabel('Width h of the MLPs')
plt.ylabel('KL divergence')
plt.title('Scaling law on the number of layer (l) and the width of the MLPs (h)')
plt.legend()
plt.show()"""

"""# print the average h over layers
data_para: pd.DataFrame = pd.read_csv('scaling_para_new.csv')
for h in [0, 100]:
    group = data_para[data_para['h'] == h].groupby(['para_head'])
    loss_group = group.mean()['loss']
    X = [h for h in loss_group.index.to_list()]
    Z = loss_group.to_list()
    plt.plot(X, Z, label=f'h={h}')
plt.xlabel('Number of parallel heads')
plt.ylabel('KL divergence')
plt.title('Scaling law on the number of parallel heads, with and without MLPs')
plt.legend()
plt.show()"""

"""# print the average d over N
data_d: pd.DataFrame = pd.read_csv('scaling_d_new.csv')
for N in [100]:
    group = data_d[data_d['N'] == N].groupby(['d'])
    loss_group = group.mean()['loss']
    X = [d for d in loss_group.index.to_list()]
    Z = loss_group.to_list()
    #minZ = min(Z)
    #Z = [z-minZ for z in Z]
    plt.plot(X, Z, label=f'N={N}')
    plt.xlabel('d')
    plt.ylabel('KL divergence') #(rescaled)')
plt.title('Scaling law on the residual dimension (d) and the number of tokens (N)')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.show()"""