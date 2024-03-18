import torch as t
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


N = 5
d = 2
max_seq_len = 21
assert max_seq_len % 3 == 0

lamb = 1.5
pi = t.softmax(lamb*t.randn(N**3), dim=0)
cat = t.distributions.Categorical(pi)


def best_loss(pi, N):
    pi = t.reshape(pi, (N, N, N))
    
    pi1 = pi.sum((1,2), keepdim=True)
    pi2 = pi.sum(2, keepdim=True)
    pi3 = pi

    entropy1 = (pi1*t.log(pi1)).sum()
    entropy2 = (pi2*t.log(pi2/pi1)).sum()
    entropy3 = (pi3*t.log(pi3/pi2)).sum()

    return -(entropy1 + entropy2 + entropy3)/3


class Transformer(t.nn.Module):
    def __init__(self, d, N, nb_layer, max_seq_len, pi):
        self.d = d
        self.N = N
        self.nb_layer = nb_layer
        self.max_seq_len = max_seq_len
        self.pi = pi

        super().__init__()

        self.word_emb = t.nn.Linear(d, N, bias=False)
        self.pos_emb = t.nn.Linear(d, max_seq_len, bias=False)
        self.seq = t.nn.Sequential(*[t.nn.MultiheadAttention(d*N**2, N**2) for _ in range(nb_layer)])
        self.unemb = t.nn.Linear(d, N, bias=False)

    def forward(self, x):
        assert x.shape[1] <= self.max_seq_len
        vect = self.word_emb.weight[x] + self.pos_emb.weight[:x.shape[1]].unsqueeze(0)

        for module in self.seq:
            vect = t.concat([vect for _ in range(self.N**2)], dim=2)
            vect = module(vect, vect, vect)[0]
            vect = t.as_strided(
                vect,
                (vect.shape[0], vect.shape[1], self.d, self.N**2),
                (vect.stride()[0], vect.stride()[1], self.d, 1)
                ).sum(-1)
            
        logits = self.unemb(vect)
        return logits
    


def generate_data(batch_size, num_batch, max_seq_len, N, cat=cat):
    
    tokens = cat.sample((batch_size*num_batch, max_seq_len//3))
    t1 = tokens%N
    t2 = (tokens//N)%N
    t3 = tokens//(N**2)
    tokens = t.concat([t1.unsqueeze(1), t2.unsqueeze(1), t3.unsqueeze(1)], dim=1)
    tokens = t.transpose(tokens, 0, 2)
    tokens = t.concat([*tokens], dim=0)
    tokens = t.transpose(tokens, 0, 1)

    dataloader = t.utils.data.DataLoader(
        tokens,
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader


def CrossEntropy(model, input, proba, pi):
    pi = t.reshape(pi, (model.N, model.N, model.N))
    nb_elements = proba.shape[0]*proba.shape[1]

    pi1 = pi.sum((1,2), keepdim=True)
    pi2 = pi.sum(2, keepdim=True)
    pi3 = pi

    t1 = input[:, t.arange(0, model.max_seq_len, 3)].flatten()
    t2 = input[:, t.arange(1, model.max_seq_len, 3)].flatten()

    prob1 = (pi2/pi1)[t1]
    prob2 = (pi3/pi2)[t1, t2]
    prob3 = pi1
    
    loss1 = t.log(proba[:, t.arange(0, model.max_seq_len, 3)].flatten(0,1))*prob1.squeeze()
    loss2 = t.log(proba[:, t.arange(1, model.max_seq_len, 3)].flatten(0,1))*prob2.squeeze()
    loss3 = t.log(proba[:, t.arange(2, model.max_seq_len-1, 3)].flatten(0,1))*prob3.squeeze()

    return -(loss1.sum() + loss2.sum() + loss3.sum())/nb_elements



def train(model, lr=1e-3, epochs=5, batch_size=2**9, num_batch=200, train_pi=False):
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    dataloader = generate_data(batch_size, num_batch, model.max_seq_len, model.N)
    b_loss = best_loss(model.pi, model.N)

    Loss = []
    th_Loss = []
    for _ in tqdm(range(epochs)):

        for batch in dataloader:

            model_logits = model(batch[:, :-1])
            model_proba = t.softmax(model_logits, dim=-1)
            next_token = batch[:, 1:]
            next_token_proba = t.zeros(next_token.shape + (model.N,))
            next_token_proba[:, next_token] += 1

            if train_pi:
                loss = CrossEntropy(model, batch[:, :-1], model_proba, model.pi)
            else:
                loss = -(t.log(model_proba)*next_token_proba).sum(-1).mean()
                
                th_loss = CrossEntropy(model, batch[:, :-1], model_proba, model.pi)
                th_Loss.append((th_loss/b_loss-1).item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            Loss.append((loss/b_loss-1).item())
    
    return {'Loss': Loss, 'th_Loss': th_Loss}


model = Transformer(d, N, 1, max_seq_len, pi)
dict = train(model, train_pi=True)

plt.plot(dict['Loss'], label='dist pred')

model = Transformer(d, N, 1, max_seq_len, pi)
dict = train(model)

plt.plot(dict['Loss'], label='next token pred')
plt.plot(dict['th_Loss'], label='next token dist pred')
plt.show()