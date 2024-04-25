import torch as t
from typing import List
from torch.utils.data import DataLoader, TensorDataset

def layer_norm(x: t.Tensor, eps=1e-10) -> t.Tensor:
    """We norm x along the last dimension."""
    mean_x = x.mean(dim=-1, keepdim=True)
    var_x = ((x-mean_x)**2).mean(dim=-1, keepdim=True)
    return (x-mean_x)/(t.sqrt(var_x)+eps)


def generate_data(batch_size: int, num_batch: int, pi: List[t.Tensor], context_window: int, one_extra=False) -> DataLoader:
    """
    Generate data using a k-states markov chain given by the distribution pi.
    Each token i for i<n_gram is given by pi[i][previous_tokens],
    Each token i for i>=n_gram is given by pi[n_gram-1][n_gram_previous_tokens].

    If one_extra = True: computes on extra token, representing the next_token of the last example in the sentences.
    """

    n_gram = len(pi)
    assert context_window >= n_gram

    extra = 1 if one_extra else 0

    token_list: List[t.Tensor] = []
    for i in range(n_gram):
        if i == 0:
            token = t.multinomial(pi[0], batch_size*num_batch, replacement=True).squeeze()
            token_list.append(token)
        elif i < n_gram-1:
            token = t.multinomial(pi[i][*token_list, :], 1).squeeze()
            token_list.append(token)
        elif i == n_gram-1:
            for _ in range(context_window-n_gram+1+extra):
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


def generate_each(pi: List[t.Tensor], eps: float = 1e-6) -> t.Tensor:
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

    return t.tensor(tokens, dtype=t.int)


def entropy(pi: List[t.Tensor], eps=1e-10) -> t.Tensor:
    "Computes the entropy of a weighted batch of distributions."
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