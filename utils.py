import torch as t
from typing import List
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
    """

    n_gram = len(pi)
    assert context_window >= n_gram

    token_list: List[t.Tensor] = []
    for i in range(n_gram):
        if i == 0:
            if batch_size*num_batch == 1:
                token = t.multinomial(pi[0], 1).unsqueeze(1)
            else:
                token = t.multinomial(pi[0], batch_size*num_batch, replacement=True).unsqueeze(1) #size: [batch_size*num_batch, 1]
            token_list.append(token) #size: [[1, batch_size*num_batch]]

        elif i == 1:
            if batch_size*num_batch == 1:
                token = t.multinomial(pi[i][token_list[0].squeeze()], 1).unsqueeze(0)
            else:
                token = t.multinomial(pi[i][token_list[0].squeeze()], 1) #size: [batch_size*num_batch, 1]
            token_list.append(token) #size: [[1, batch_size*num_batch], [1, batch_size*num_batch]]

        elif i < n_gram-1:
            token_tensor = t.cat(token_list, dim=1) #size: [batch_size*num_batch, i]
            cond_pi_list = []
            for token_l in token_tensor:
                py = pi[i] #size: [N]*(i+1)
                for tok in token_l: #size: i
                    py = py[tok].squeeze()
                cond_pi_list.append(py.unsqueeze(0))
            cond_pi = t.cat(cond_pi_list, dim=0) #size: [i, N]
            token = t.multinomial(cond_pi, 1) #size: [i, 1]
            token_list.append(token)

        elif i == n_gram-1:
            for _ in range(context_window-n_gram+1):
                token_tensor = t.cat(token_list[-n_gram+1:], dim=1) #size: [batch_size*num_batch, n_gram-1]
                cond_pi_list = []
                for token_l in token_tensor:
                    py = pi[i]
                    for tok in token_l:
                        py = py[tok].squeeze()
                    cond_pi_list.append(py.unsqueeze(0))
                cond_pi = t.cat(cond_pi_list, dim=0) #size: [i, N]
                token = t.multinomial(cond_pi, 1) #size: [i, 1]
                token_list.append(token)
    
    token_list = [token.squeeze().unsqueeze(-1) for token in token_list]
    tokens = t.cat(token_list, dim=-1)
    dataset = TensorDataset(tokens)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader


def entropy(pi: List[t.Tensor], eps=1e-10) -> t.Tensor:
    """Computes the entropy of a weighted batch of distributions."""
    ent = -t.log(pi[-1]+eps)
    for j in range(len(pi)-1, -1, -1):
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