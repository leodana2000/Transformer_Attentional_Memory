import torch as t
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import Dict

N = 5
d = 2

lamb = 7
pi = t.softmax(lamb*t.randn(N**3), dim=0)

### trigram with MLPs

def best_loss(params: Dict):
    """Computes the best possible loss for the n_gram problem."""
    pi = params['pi']
    N = params['N']

    pi_1 = t.reshape(pi, (N,)*3)
    pi_2 = pi_1.sum(-1, keepdim=True)
    entropy = -(pi_1*t.log(pi_1/pi_2)).sum()

    return entropy


class MLP(t.nn.Module):
    def __init__(self, d: int, N: int, h: int, nb_layers: int, pi: t.Tensor):
        self.meta_params: Dict = {
            'd': d,
            'N': N,
            'h': h,
            'pi': pi,
            'nb_layers': nb_layers,
        }

        super().__init__()

        self.word_emb = t.nn.Linear(d, N, bias=False)
        self.rotation = t.nn.Linear(d, d, bias=False)
        self.mlp = t.nn.Sequential(*[
            t.nn.Sequential(*[
                t.nn.Linear(d,h),
                t.nn.ReLU(),
                t.nn.Linear(h,d),
            ]) for _ in range(nb_layers)
        ])
        self.unemb = t.nn.Linear(d, N, bias=False)

    def forward(self, t1, t2):
        v1 = self.word_emb.weight[t1]
        v2 = self.word_emb.weight[t2]
        v = v2 + self.rotation(v1)
        for mlp in self.mlp:
            v = mlp(v)+v
        return self.unemb(v)
    
    def all_path(self, t1, t2):
        assert len(self.mlp) == 2
        v1 = self.word_emb.weight[t1]
        v2 = self.word_emb.weight[t2]
        v = v2 + self.rotation(v1)
        mlp0 = self.mlp[0]
        mlp1 = self.mlp[1]
        L = [v, mlp0(v)+v, mlp0(v), mlp1(v), mlp1(v+mlp0(v)), mlp0(v)+v+mlp1(v), mlp0(v)+v+mlp1(mlp0(v)), v+mlp1(mlp0(v)+v), v+mlp1(v), v+mlp1(mlp0(v)), mlp0(v)+mlp1(mlp0(v)+v), mlp0(v)+mlp1(mlp0(v)), mlp0(v)+mlp1(v), mlp0(v)+v+mlp1(mlp0(v)+v)]
        return [self.unemb(v) for v in L]



def CrossEntropy(params: Dict, input: t.Tensor, proba: t.Tensor):
    """Computes the cross-entropy in the case where we know the target distribution."""
    pi = params['pi']
    N = params['N']

    pi_1 = t.reshape(pi, (N**2, N))

    loss = -(pi_1*t.log(proba)).sum()
    return loss


def generate_data(num_batch: int, params: Dict):
    """Generates num_batch batach containing all the different sentences possible."""
    N = params['N']

    tokens = t.arange(0, N**2, 1)
    tokens = t.concat([(tokens.unsqueeze(1)//(N**i))%N for i in range(2)], dim=1)
    tokens = t.concat([tokens.unsqueeze(0)]*num_batch, dim=0)

    return tokens


def train(model: MLP, lr=1e-4, epochs=450, num_batch=50):
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    dataloader = generate_data(num_batch, model.meta_params)
    b_loss = best_loss(model.meta_params)

    Loss = []
    for _ in tqdm(range(epochs)):

        for batch in dataloader:
            model_logits = model(batch[:, 0], batch[:, 1])
            model_proba = t.softmax(model_logits, dim=-1)

            loss = CrossEntropy(model.meta_params, batch, model_proba)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            Loss.append((loss/b_loss-1).item())
    
    return {'Loss': Loss}



def show_tokens(model: MLP):
    tokens = generate_data(1, model.meta_params)
    t1 = tokens[0, :, 0]
    t2 = tokens[0, :, 1]
    v1 = model.word_emb.weight[t1]
    v2 = model.word_emb.weight[t2]
    V = v2 + model.rotation(v1)
    for v in V:
        plt.plot([0, v[0].item()], [0, v[1].item()], color='C0')

    for w, b in zip(model.mlp[0][0].weight, model.mlp[0][0].bias):
        mean = b*w/(t.norm(w)**2)
        normed = w/t.norm(w)
        orth_normed = t.Tensor([normed[1], -normed[0]])
        ratio = 20
        plt.plot([mean[0].item(), (mean+normed/ratio)[0].item()], [mean[1].item(), (mean+normed/ratio)[1].item()], color='C1')
        plt.plot([(mean-orth_normed/ratio)[0].item(), (mean+orth_normed/ratio)[0].item()], [(mean-orth_normed/ratio)[1].item(), (mean+orth_normed/ratio)[1].item()], color='C1')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()



"""model = MLP(d, N, 2*N**2, 1, pi)
dict = train(model)
plt.plot(dict['Loss'], label=f'{2*N**2}')
model = MLP(d, N, 2*N, 1, pi)
dict = train(model)
plt.plot(dict['Loss'], label=f'{2*N}')
model = MLP(d, N, N**2, 2, pi)
dict = train(model)
plt.plot(dict['Loss'], label=f'double {N**2}')
model = MLP(d, N, N, 2, pi)
dict = train(model)
plt.plot(dict['Loss'], label=f'double {N}')
plt.legend()
plt.show()"""


"""model = MLP(d, N, N**2, 1, pi)
show_tokens(model)
dict = train(model)
show_tokens(model)"""


def show_hidden(model: MLP):
    tokens = generate_data(1, model.meta_params)
    t1 = tokens[0, :, 0]
    t2 = tokens[0, :, 1]
    v1 = model.word_emb.weight[t1]
    v2 = model.word_emb.weight[t2]
    v = v2 + model.rotation(v1)
    v_h = model.mlp[0][0](v)
    print(v.shape, v_h.shape)
    r_v_h = t.nn.ReLU()(v_h)
    print((r_v_h>0).to(t.float).sum(-1))


"""model = MLP(d, N, N**2, 1, pi)
show_hidden(model)
dict = train(model)
show_hidden(model)"""


model=MLP(d, N, N**2, 2, pi)
train(model)
tokens = generate_data(1, model.meta_params)
t1 = tokens[0, :, 0]
t2 = tokens[0, :, 1]
probas = [t.softmax(u, dim=-1) for u in model.all_path(t1, t2)]
loss = [CrossEntropy(model.meta_params, None, proba).item() for proba in probas]
L = ["v", "mlp0(v)+v", "mlp1(v)", "mlp1(v+mlp0(v))", "mlp0(v)+v+mlp1(v)", "mlp0(v)+v+mlp1(mlp0(v))", "v+mlp1(mlp0(v)+v)", "v+mlp1(v)", "v+mlp1(mlp0(v))", "mlp0(v)+mlp1(mlp0(v)+v)", "mlp0(v)+mlp1(mlp0(v))"," mlp0(v)+mlp1(v)", "mlp0(v)+v+mlp1(mlp0(v)+v)"]
for text, loss in zip(L, loss):
    print(text, loss)