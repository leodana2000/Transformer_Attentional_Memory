import torch as t
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Dict

n_gram = 3
N = 5
d = 2
max_seq_len = n_gram
assert max_seq_len % n_gram == 0

lamb = 1.5
pi = t.softmax(lamb*t.randn(N**n_gram), dim=0)


def best_loss(params: Dict):
    """Computes the best possible loss for the n_gram problem."""
    pi = params['pi']
    N = params['N']
    n_gram = params['n_gram']

    pi_1 = t.reshape(pi, (N,)*n_gram)

    entropy = 0
    for _ in range(n_gram):
        pi_2 = pi_1.squeeze()
        pi_1 = pi_2.sum(-1, keepdim=True)
        entropy -= (pi_2*t.log(pi_2/pi_1)).sum()

    return entropy/n_gram


class Transformer(t.nn.Module):
    def __init__(self, d: int, N: int, nb_layer: int, max_seq_len: int, n_gram: int, pi: t.Tensor):
        self.meta_params: Dict = {
            'd': d,
            'N': N,
            'n_gram': n_gram,
            'max_seq_len': max_seq_len,
            'pi': pi,
            'nb_layer': nb_layer,
        }

        super().__init__()

        self.word_emb = t.nn.Linear(d, N, bias=False)
        self.pos_emb = t.nn.Linear(d, max_seq_len, bias=False)
        self.seq = t.nn.Sequential(*[t.nn.MultiheadAttention(d*N**2, N**2) for _ in range(nb_layer)])
        self.unemb = t.nn.Linear(d, N, bias=False)

    def forward(self, x: t.Tensor):
        assert x.shape[1] <= self.meta_params['max_seq_len']
        d = self.meta_params['d']
        N = self.meta_params['N']

        vect = self.word_emb.weight[x] + self.pos_emb.weight[:x.shape[1]].unsqueeze(0)

        for module in self.seq:
            vect = t.concat([vect for _ in range(N**2)], dim=2)
            vect = module(vect, vect, vect)[0]
            vect = t.as_strided(
                vect,
                (vect.shape[0], vect.shape[1], d, N**2),
                (vect.stride()[0], vect.stride()[1], d, 1),
                ).sum(-1)
            
        logits = self.unemb(vect)
        return logits
    


def generate_data(batch_size: int, num_batch: int, params: Dict, unif=False):
    """Generate batch_size*num_batch sequences using the n_gram distribution."""
    max_seq_len=params['max_seq_len']
    N=params['N']
    n_gram=params['n_gram']
    pi: t.Tensor =params['pi']
    if unif:
        pi = t.softmax(t.zeros_like(pi), dim=0)
    
    cat = t.distributions.Categorical(pi)
    tokens = cat.sample((batch_size*num_batch, max_seq_len//n_gram))
    tokens = t.concat([(tokens.unsqueeze(1)//(N**i))%N for i in range(n_gram)], dim=1)
    tokens = t.transpose(tokens, 0, 2)
    tokens = t.concat([*tokens], dim=0)
    tokens = t.transpose(tokens, 0, 1)

    dataloader = t.utils.data.DataLoader(
        tokens,
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader


def exhaustive_data(num_batch: int, params: Dict):
    """Generates num_batch batach containing all the different sentences possible."""
    n_gram=params['n_gram']
    max_seq_len=params['max_seq_len']

    tokens = t.arange(0, N**max_seq_len, 1)
    tokens = t.concat([(tokens.unsqueeze(1)//(N**i))%N for i in range(n_gram)], dim=1)
    tokens = t.concat([tokens.unsqueeze(0)]*num_batch, dim=0)

    return tokens


def CrossEntropy(params: Dict, input: t.Tensor, proba: t.Tensor): #takes longer to execute than it should!!
    """Computes the cross-entropy in the case where we know the target distribution."""
    max_seq_len=params['max_seq_len']
    pi = params['pi']
    N = params['N']
    n_gram = params['n_gram']

    pi_1 = t.reshape(pi, (N,)*n_gram)

    loss = 0
    tokens = t.as_strided(input, (n_gram, input.shape[0], max_seq_len//n_gram), (1, max_seq_len, n_gram))[:-1].flatten(1,2)
    logits = t.log(t.as_strided(proba, (n_gram, proba.shape[0], max_seq_len//n_gram), (1, max_seq_len, n_gram))).flatten(1,2).unsqueeze(-1)

    target_dist = []
    for i in range(1, n_gram):
        pi_2 = pi_1.squeeze() #distribution of token up to n_gram-i+1
        pi_1 = pi_2.sum(-1, keepdim=True) #distribution of token up to n_gram-i
        target_dist.append((pi_2/pi_1)[*tokens[:(n_gram-i)]].squeeze().unsqueeze(0)) #distribution of tokens n_gram-i+1 knowing n_gram-i

    target_dist.append(pi_1.squeeze().unsqueeze(0).unsqueeze(0)[:, t.zeros(input.shape[0]*max_seq_len//n_gram, dtype=t.int), :]) #distribution of the first tokens, pi_1 is constant
    target_proba: t.Tensor = t.cat(target_dist, dim=0)
    loss -= (target_proba*logits).sum()

    nb_elements = proba.shape[0]*proba.shape[1]
    return loss/nb_elements



def train(model: Transformer, lr=1e-4, epochs=5, batch_size=2**8, num_batch=200, train_pi=False, opt_batch=False):
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    if opt_batch and (model.meta_params['n_gram'] == model.meta_params['max_seq_len']):
        dataloader = exhaustive_data(num_batch, model.meta_params)
    else:
        dataloader = generate_data(batch_size, num_batch, model.meta_params, unif=train_pi)
    b_loss = best_loss(model.meta_params)

    Loss = []
    th_Loss = []
    for _ in tqdm(range(epochs)):

        for batch in dataloader:
            model_logits = model(batch[:, :-1])
            model_proba = t.softmax(model_logits, dim=-1)

            if train_pi:
                loss = CrossEntropy(model.meta_params, batch[:, :-1], model_proba)
            else:
                next_token = batch[:, 1:]
                next_token_proba = t.zeros((next_token.shape[1], next_token.shape[0], model.meta_params['N']))
                next_token_proba[:, next_token] += 1
                next_token_proba = t.transpose(next_token_proba, 0, 1)

                loss = -(t.log(model_proba)*next_token_proba).sum(-1).mean()
                
                th_loss = CrossEntropy(model.meta_params, batch[:, :-1], model_proba)
                th_Loss.append((th_loss/b_loss-1).item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            Loss.append((loss/b_loss-1).item())
    
    return {'Loss': Loss, 'th_Loss': th_Loss}


model = Transformer(d, N, 1, max_seq_len, n_gram, pi)
dict = train(model, train_pi=True)

plt.plot(dict['Loss'], label='train_pi')

model = Transformer(d, N, 1, max_seq_len, n_gram, pi)
dict = train(model)

plt.plot(dict['th_Loss'], label='base')

model = Transformer(d, N, 1, max_seq_len, n_gram, pi)
dict = train(model, train_pi=True, opt_batch=True)

plt.plot(dict['Loss'], label='opt_batch')
plt.show()