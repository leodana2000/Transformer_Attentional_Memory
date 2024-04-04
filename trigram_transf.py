import torch as t
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Dict
import pandas as pd
from time import time

n_gram = 3
N = 100
d = 12
h = 10*d
nb_layers = 3
nb_head = 4
max_seq_len = n_gram
assert max_seq_len >= n_gram
assert n_gram == 3

lamb = 5.
t.manual_seed(0)
pi = t.softmax(lamb*t.randn(N**n_gram), dim=0)


class Transformer(t.nn.Module):
    """Transformer architecture with parallel attention heads and MLPs, additive positional embeddings, and layer-norm."""
    def __init__(self, d: int, N: int, h: int, nb_head: int, nb_layers: int, max_seq_len: int, n_gram: int, pi: t.Tensor, parallel_heads: int = 1):
        assert d%nb_head == 0
        self.meta_params: Dict = {
            'd': d,
            'N': N,
            'h': h,
            'n_gram': n_gram,
            'nb_head': nb_head,
            'max_seq_len': max_seq_len,
            'pi': pi,
            'nb_layers': nb_layers,
        }

        super().__init__()

        self.word_emb = t.nn.Linear(d, N, bias=False)
        self.pos_emb = t.nn.Linear(d, max_seq_len, bias=False)
        self.attn_seq = t.nn.Sequential(
            *[
                t.nn.Sequential(
                    *[t.nn.MultiheadAttention(d, nb_head, batch_first=True) for _ in range(parallel_heads)]
                )
            for _ in range(nb_layers)]
        )
        self.mlp_seq = t.nn.Sequential(
            *[t.nn.Sequential(
                *[t.nn.Linear(d, h, bias=True), t.nn.GELU(), t.nn.Linear(h, d, bias=False)]
            ) for _ in range(nb_layers)]
        )
        self.unemb = t.nn.Linear(d, N, bias=False)


    def layer_norm(self, x, eps=1e-6):
        """We norm x along the last dimension."""
        mean_x = x.mean(dim=-1, keepdim=True)
        var_x = ((x-mean_x)**2).mean(dim=-1, keepdim=True)
        return (x-mean_x)/(t.sqrt(var_x)+eps)


    def forward(self, x: t.Tensor):
        assert x.shape[1] <= self.meta_params['max_seq_len']
        attn_mask = t.tril(t.ones((x.shape[1], x.shape[1]))) == 0

        res = self.word_emb.weight[x]
        pos = self.pos_emb.weight[:x.shape[1]].unsqueeze(0)
        for para_attn, mlp in zip(self.attn_seq, self.mlp_seq):
            norm_res = self.layer_norm(res) #we add the positional embedding at each layer
            para_res = 0.
            for attn in para_attn: #if there is parallel attention, each mechanism is computed in parallel and then added in the stream
                para_res += attn(norm_res+pos, norm_res+pos, norm_res+pos, attn_mask=attn_mask)[0]
            res = mlp(norm_res) + res + para_res #we also compute mlp in parallel of attention
            
        logits = self.unemb(res) #no layer-norm so the transformer can modulate its confidence
        return logits
    

def generate_data(batch_size: int, num_batch: int, params: Dict): #change so that it can generate markov-trigram
    """Generate batch_size*num_batch sequences using the n_gram distribution."""
    max_seq_len=params['max_seq_len']
    N=params['N']
    n_gram=params['n_gram']
    pi: t.Tensor = params['pi']
    
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


def entropy(params):
    pi=params['pi']
    N=params['N']
    pi = t.reshape(pi, (N**2, N))
    p = pi/pi.sum(-1, keepdim=True)
    return -(pi.sum(-1)*(p*t.log(p)).sum(-1)).sum()


def to_target(tokens, N):
    batch_size, max_seq_len = tokens.shape
    target_tokens = t.zeros(batch_size, max_seq_len, N)
    target_tokens[:, :] = t.arange(N)
    target_tokens = (target_tokens == tokens.unsqueeze(-1)).to(t.float)
    return target_tokens


def train(model: Transformer, lr=1e-3, epochs=1, batch_size=2**10, num_batch=5000):
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    dataloader = generate_data(batch_size, num_batch, model.meta_params)
    ent = entropy(model.meta_params)

    Loss = []
    for _ in range(epochs):

        for batch in dataloader:
            model_logits = model(batch)
            model_proba = t.softmax(model_logits, dim=-1)

            target_tokens = to_target(batch, model.meta_params['N'])
            loss = -ent-(target_tokens[:, 2:, :]*t.log(model_proba[:, 1:-1, :])).sum(-1).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            Loss.append(loss.item())
    
    return {'Loss': Loss}


model = Transformer(d, N, 100, nb_head, nb_layers, max_seq_len, n_gram, pi)
dict = train(model)
plt.plot(dict['Loss'])

model = Transformer(d, N, 1000, nb_head, nb_layers, max_seq_len, n_gram, pi)
dict = train(model)
plt.plot(dict['Loss'])

plt.show()

"""
loss=[]
lamb_list=[]
N_list=[]
d_list=[]
h_list=[]
nb_layers_list=[]
nb_head_list=[]
para_list=[]

count = 0
max_count = 0
for N in [50]:
    t.manual_seed(0)
    logits = t.randn(N**3)
    for lamb in [2.]:
        pi = t.softmax(lamb*logits, dim=0)
        for d in [5]:
            for h in [100]:
                for nb_head in [1]:
                    for nb_layers in [1]:
                        for parallel_heads in [1]:
                            N_list.append(N)
                            d_list.append(d)
                            h_list.append(h)
                            nb_head_list.append(nb_head)
                            nb_layers_list.append(nb_layers)
                            para_list.append(parallel_heads)

                            model = Transformer(d, N, h, nb_head, nb_layers, 3, 3, pi, parallel_heads=parallel_heads)
                            lamb_list.append(entropy(model.meta_params).item())
                            dict = train(model, lr=1e-3, epochs=10, batch_size=2**10, num_batch=200)
                            loss.append(sum(dict['Loss'][-100:-1])/100)

                            count+=1
                            print(count/max_count)

dict={
    'lamb': lamb_list,
    'N': N_list,
    'd': d_list,
    'h': h_list,
    'nb_layers': nb_layers_list,
    'nb_head': nb_head_list,
    'para_head': para_list,
    'loss': loss,
}

data = pd.DataFrame(dict)
data.to_csv('.csv', index=False)
"""