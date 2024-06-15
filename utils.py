import torch as t
import numpy as np
from typing import List, Tuple, Any
from torch.utils.data import DataLoader, TensorDataset

def layer_norm(x: t.Tensor, eps=1e-10) -> t.Tensor:
    """We norm x along the last dimension."""
    mean_x = x.mean(dim=-1, keepdim=True)
    var_x = ((x-mean_x)**2).mean(dim=-1, keepdim=True)
    return (x-mean_x)/(t.sqrt(var_x)+eps)


def compute_div(W_E: t.Tensor, W_U: t.Tensor, pi: List[t.Tensor]):
    with t.no_grad():
            f = t.log(t.softmax(W_E@W_U.mH, dim=-1))
            PI = t.transpose(pi[2], 0, 1).flatten(0, 1) #be carefull to transpose !!! first dim is +N
            L = t.log(PI)
            prior = (pi[0].unsqueeze(-1)*pi[1]).flatten(0, 1)
            return ((PI*(L-f)).sum(-1)*prior).sum()


def generate_data(batch_size: int, num_batch: int, pi: List[t.Tensor], context_window: int) -> DataLoader:
    """
    Generate data using a k-states markov chain given by the distribution pi.
    Each token i for i<n_gram is given by pi[i][previous_tokens],
    Each token i for i>=n_gram is given by pi[n_gram-1][n_gram_previous_tokens].

    If one_extra = True: computes on extra token, representing the next_token of the last example in the sentences.
    """

    n_gram = len(pi)
    assert context_window >= n_gram

    token_list: List[t.Tensor] = []
    for i in range(n_gram):
        if i == 0:
            token = t.multinomial(pi[0], batch_size*num_batch, replacement=True).squeeze()
            token_list.append(token)
        elif i < n_gram-1:
            token = t.multinomial(pi[i][*token_list, :], 1).squeeze()
            token_list.append(token)
        elif i == n_gram-1:
            for _ in range(context_window-n_gram+1):
                token = t.multinomial(pi[n_gram-1][*token_list[-n_gram+1:], :], 1).squeeze()
                token_list.append(token)
    
    token_list = [token.unsqueeze(-1) for token in token_list]
    tokens = t.cat(token_list, dim=-1)
    dataset = TensorDataset(tokens)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader


def generate_each(pi: List[t.Tensor], eps: float = 1e-10) -> t.Tensor:
    """Generate every sequences of tokens that have more than probability eps."""
    n_gram = len(pi)
    N = pi[0].shape[0]

    tokens = []
    for i in range(N**n_gram):
        is_non_zero = True
        indices = [(i//(N**j))%N for j in range(n_gram)]
        cumulative_proba = 1.
        ind = 0
        while (ind < n_gram) and is_non_zero:
            cumulative_proba *= pi[ind][*indices[:ind+1]].item()
            ind += 1
            is_non_zero = is_non_zero and (cumulative_proba > eps)
        if is_non_zero:
            tokens.append(indices)
    return t.tensor(tokens, dtype=t.long)


def generate_uniform_simplex(nb_points, add_special=True) -> List[Tuple[Any, Any, Any]]:
    """
    Generates nb_points uniformly on the 2-simplex. 
    Add extra points at the tip and center of the simplex if 'add_special=True.'
    """

    U, V = list(np.random.rand(nb_points)), list(np.random.rand(nb_points))
    if add_special:
        U += [0, 1, 1, 4/9]
        V += [0, 0, 1, 1/2]

    data=[]
    for u, v in zip(U, V):
        sqrt_u = np.sqrt(u)
        p, q, r = 1-sqrt_u, sqrt_u*(1-v), sqrt_u*v
        data.append((p, q, r))
    
    return data


def generate_uniform_sphere(nb_points) -> List[Tuple[Any, Any, Any]]:
    """Generates nb_points uniformly on the sphere."""
    data_tensor = t.randn((3, nb_points))
    data_tensor = data_tensor/t.norm(data_tensor, dim=0, keepdim=True)
    data = np.array(data_tensor).tolist()
    return data


def entropy(pi: List[t.Tensor], eps=1e-10) -> t.Tensor:
    """Computes the entropy of a weighted batch of distributions."""
    ent = -t.log(pi[-1]+eps)
    for j in range(len(pi)-1, 0-1, -1):
        ent = (pi[j]*ent).sum(-1)
    return ent


def power_unif_law(alphas: List[float], nb_tokens: List[int], N: int) -> List[t.Tensor]:
    """
    Generate a distribution for an n_gram. Each conditional distribution has nb_tokens[l] non zero probabilities, 
    which are distributed as a power law of parameter alphas[l].
    """
    pi=[]
    for i, (alpha, nb_token) in enumerate(zip(alphas,nb_tokens)):
        dist = t.Tensor([alpha**i for i in range(N)])
        dist[nb_token:] = 0
        dist = dist/dist.sum()
        pi.append(t.cat([dist[t.randperm(N)] for _ in range(N**i)], dim=0).reshape((N,)*(i+1)))
    return pi


def first_position_law(nb_tokens: List[int], N: int, d: int, feature_importance=1.) -> List[t.Tensor]:
    """
    Generate a distribution which is random on the first d position, and uniform on the rest.
    """
    pi=[]
    for i, nb_token in enumerate(nb_tokens):
        dist = t.Tensor([1/N for i in range(N)])
        dist[nb_token:] = 0
        dist = dist/dist.sum()
        pi.append(t.cat([dist[t.randperm(N)] for _ in range(N**i)], dim=0).reshape((N,)*(i+1)))

    dists = []
    mask = t.zeros(N)
    mask[:d] = 1
    for _ in range(N**len(nb_tokens)):
        logits = t.randn(N)*mask*t.Tensor([feature_importance**i for i in range(N)])
        dists.append(t.softmax(logits, dim=0).unsqueeze(0))
    pi.append(t.cat(dists, dim=0).reshape((N,)*(len(nb_tokens)+1)))
    return pi


def last_position_law(nb_tokens: List[int], N: int, k: int, p:float) -> List[t.Tensor]:
    """
    Generate a distribution which has probability p on one of the tokens, and uniform otherwise.
    The token which has probability p is chosen randomly with a meta-probability scaling as a Zipf law of power k.
    The priori distribution is always uniform.
    """

    pi=[]
    for i, nb_token in enumerate(nb_tokens):
        dist = t.Tensor([1/N for i in range(N)])
        dist[nb_token:] = 0
        dist = dist/dist.sum()
        pi.append(t.cat([dist[t.randperm(N)] for _ in range(N**i)], dim=0).reshape((N,)*(i+1)))

    dists = []
    for _ in range(N**len(nb_tokens)):
        meta = t.Tensor([(1-i/N)**k for i in range(N)])
        meta_choice = t.distributions.Categorical(probs=meta/(meta.sum()))
        choice = meta_choice.sample((1,))
        dist = t.Tensor([p if i==choice else (1-p)/(N-1) for i in range(N)])
        dists.append(dist.unsqueeze(0))
    pi.append(t.cat(dists, dim=0).reshape((N,)*(len(nb_tokens)+1)))
    return pi


def gen_d_law(nb_tokens: List[int], N: int, d: int, axis_aligned=False, feature_importance=1., flat_coef=0.1) -> Tuple[List[t.Tensor], t.Tensor]:
    """
    Generate a distribution of rank d.
    If axis_aligned=True, then the distribution is generated alog the first d coordinates.
    The priori distribution is always uniform.
    """

    pi=[]
    for i, nb_token in enumerate(nb_tokens):
        dist = t.Tensor([1/N for _ in range(N)])
        dist[nb_token:] = 0
        dist = dist/dist.sum()
        pi.append(t.cat([dist[t.randperm(N)] for _ in range(N**i)], dim=0).reshape((N,)*(i+1)))

    directions = t.nn.Linear(d, N).requires_grad_(False)
    if axis_aligned:
        directions.weight = t.nn.Parameter(t.eye(N)[:, :d]).requires_grad_(False)
    else:
        directions.weight = t.nn.Parameter(t.randn((N, N))[:, :d]).requires_grad_(False)

    dists=[]
    for _ in range(N**len(nb_tokens)):
        logits = directions(t.randn(d)*t.Tensor([feature_importance**i for i in range(d)]))
        dists.append(t.softmax(logits*flat_coef, dim=-1).unsqueeze(0))
    pi.append(t.cat(dists, dim=0).reshape((N,)*(len(nb_tokens)+1)))
    return pi, directions.weight


def almost_rank_d(nb_tokens: List[int], N: int, d: int, axis_aligned=False, eig_high=1., eig_low=0.1) -> List[t.Tensor]:
    """
    Generate a distribution which has almost rank d.
    If axis_aligned=True, then the distribution is generated alog the first d coordinates.
    The priori distribution is always uniform.
    """
    n_gram = len(nb_tokens)+1

    pi=[]
    for i, nb_token in enumerate(nb_tokens):
        dist = t.Tensor([1/N for _ in range(N)])
        dist[nb_token:] = 0
        dist = dist/dist.sum()
        pi.append(t.cat([dist[t.randperm(N)] for _ in range(N**i)], dim=0).reshape((N,)*(i+1)))

    logits = t.randn((N**(n_gram-1), N))
    U, eig, V = t.linalg.svd(logits)
    eig = t.Tensor([eig_high if i<d else eig_low for i in range(N)])
    if axis_aligned:
        V = t.eye(N)
    logits = t.einsum('MN, N, Nn -> Mn', U[:, :N], eig, V)

    pi.append(t.softmax(logits, dim=-1).reshape((N,)*n_gram))
    return pi


#Heatmap display

import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets 
from IPython.display import display, clear_output

def plot_heatmap(map: t.Tensor, title: str, xticklabels=None, yticklabels=None):
    if xticklabels == None and yticklabels == None:
        sns.heatmap(map, center=0, cmap='bwr')
    else:
        sns.heatmap(map, center=0, cmap='bwr', xticklabels=xticklabels, yticklabels=yticklabels)
    plt.title(title)
    plt.show()


def display_heatmaps(heatmap_list):
    num_heatmaps = len(heatmap_list)
    slider = widgets.IntSlider(value=0, min=0, max=num_heatmaps-1, step=1, description='Select Heatmap:')

    sns.heatmap(heatmap_list[0], center=0, cmap='bwr') # Display the first heatmap initially

    def update_heatmap(change):
        clear_output(wait=True)
        sns.heatmap(heatmap_list[change['new']], center=0, cmap='bwr')
        display(slider)

    slider.observe(update_heatmap, names='value')

    display(slider)