import numpy as np
import torch as t
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
We test here if the empirical loss and the order 2 approximations are close.
"""

def CrossEntropyLoss(pi_z: t.Tensor):
    def loss(pi_k: t.Tensor, pi_p: t.Tensor):
        best_loss = -(pi_z*(pi_k*t.log(pi_p)).sum(-1)).sum(0)
        return best_loss
    return loss


class AE_bigram(t.nn.Module):
    def __init__(self, d: int, pi_z: t.Tensor, pi_k: t.Tensor, train_U=True):
        super().__init__()
        N = pi_z.shape[-1]
        M = pi_k.shape[-1]

        self.W_E = t.nn.Linear(d, N, bias=False)
        self.W_U = t.nn.Linear(d, M, bias=False)
        self.W_U.requires_grad_(train_U)

        self.N: int = N
        self.M: int = M
        self.d: int = d

        self.pi_z: t.Tensor = pi_z
        self.pi_k: t.Tensor = pi_k

    def forward(self, x):
        return self.W_U(self.W_E.weight[x])


def C(b):
    B = (1-2*b-2*t.log((1-b)/(b + 1e-6) + 1e-6))/(2*b + 1e-6)
    D_b = B+t.sqrt(B**2+(1-b)/b)
    C_b = (D_b)/((1+D_b)**2)
    return C_b


def train(model: AE_bigram, lr=1e-3, epochs=15, batch_size=2**10, num_batch=3000):
    f_loss = CrossEntropyLoss(t.softmax(t.zeros((batch_size)), dim=-1))

    best_loss = CrossEntropyLoss(pi_z=model.pi_z)(model.pi_k, model.pi_k)
    cat_pi_z = t.distributions.Categorical(model.pi_z)
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    Loss = []
    for _ in tqdm(range(epochs)):
        loss_avg = 0
        count = 0

        for b in range(num_batch):
            ex = cat_pi_z.sample((batch_size, 1)).squeeze()

            model_pi = t.softmax(model(ex), dim=-1)
            true_pi = model.pi_k[ex]
            
            loss = f_loss(true_pi, model_pi)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (num_batch - b) <= 100:
                count += 1
                loss_avg += loss.item()

        Loss.append((loss_avg/count-best_loss))
    
    return {'Div': Loss}


def is_W_E_opt(model, pi_k):
    W_E = model.W_E.weight
    W_U = model.W_E.weight
    W_E_opt = t.zeros_like(W_E)

    for z in range(model.N):
        pi = pi_k[z]
        lpi = t.log(pi_k[z])

        EW_U = pi@W_U
        EW_Ulp = (pi*lpi)@W_U
        Elp = (pi@lpi)
        cov = EW_Ulp - EW_U*Elp

        EW_UW_U = t.einsum('Nd, ND, N -> dD', W_U, W_U, pi)
        var = EW_UW_U - EW_U.unsqueeze(-1)@EW_U.unsqueeze(0)
        inv_var = t.inverse(var)

        W_E_opt[z] = inv_var@cov

    return t.norm(W_E-W_E_opt).item()


def is_W_UW_U_Id(model: AE_bigram):
    W_U: t.Tensor = model.W_U.weight
    pseudo_inv = W_U@W_U.pinverse()
    return t.max(t.abs(pseudo_inv-t.eye(model.N))).item()


def Q_bound(model, pi_k, pi_z):
    x = t.arange(0, model.N, 1)
    with t.no_grad():

        logits = model(x)
        Z = logits - t.log(pi_k)
        mean = t.cat([t.linspace(minZ, maxZ, 1000).unsqueeze(0) for minZ, maxZ in zip(t.min(Z, dim=-1)[0], t.max(Z, dim=-1)[0])], dim=0)
        Z = Z.unsqueeze(-1) - mean

        pos = (Z > 0).to(t.float)
        neg = pos-1

        D_b = (1-2*pi_k-2*t.log((1-pi_k)/pi_k))/(2*pi_k)
        D_b = D_b + t.sqrt((1-pi_k)/pi_k + D_b**2)
        D_b = (D_b/((1+D_b)**2)).unsqueeze(-1)

        p_exp = t.min(t.exp(pos*Z)*(pi_k/(1-pi_k)).unsqueeze(-1), t.ones_like(t.exp(pos*Z)*(pi_k/(1-pi_k)).unsqueeze(-1)))
        C_b = (p_exp)/((1+p_exp)**2)

        bound_1 = (pi_z.unsqueeze(-1)*(pos*C_b*Z**2/2 + pi_k.unsqueeze(-1)*neg*Z).sum(1)).sum(0)
        bound_1 = t.min(bound_1)

        bound_2 = (pi_z.unsqueeze(-1)*(C_b*Z**2/2 + pi_k.unsqueeze(-1)*neg*Z).sum(1)).sum(0)
        bound_2 = t.min(bound_2)

        bound_3 = (pi_z.unsqueeze(-1)*(pos*C_b*Z**2/2 + pi_k.unsqueeze(-1)*t.abs(Z)).sum(1)).sum(0)
        bound_3 = t.min(bound_3)

        bound_4 = (pi_z.unsqueeze(-1)*(C_b*Z**2/2 + pi_k.unsqueeze(-1)*t.abs(Z)).sum(1)).sum(0)
        bound_4 = t.min(bound_4)

        bound_5 = (pi_z.unsqueeze(-1)*(p_exp*Z**2/2 + pi_k.unsqueeze(-1)*t.abs(Z)).sum(1)).sum(0)
        bound_5 = t.min(bound_5)

        bound_6 = (pi_z.unsqueeze(-1)*(pos*D_b*Z**2/2 + pi_k.unsqueeze(-1)*neg*Z).sum(1)).sum(0)
        bound_6 = t.min(bound_6)

        bound_7 = (pi_z.unsqueeze(-1)*(Z**2/8 + pi_k.unsqueeze(-1)*t.abs(Z)).sum(1)).sum(0)
        bound_7 = t.min(bound_7)
        return bound_1, bound_2, bound_3, bound_4, bound_5, bound_6, bound_7


#meta parameters
N = 100
M = 100
d = 10

#bigram distributions
l = 1
pi_z = t.softmax(l*t.randn(N), dim=-1)
pi_k = t.softmax(l*t.randn(N, M), dim=-1)

model = AE_bigram(d, pi_z, pi_k, train_U=True)
dict = train(model)
plt.plot(dict['Div'])

plt.show()


"""
Bound_1=[]
Bound_2=[]
Bound_3=[]
Bound_4=[]
Bound_5=[]
Bound_6=[]
Bound_7=[]
entropy=[]
Div=[]
for l in t.linspace(0, 4, 30):
    pi_z = t.softmax(l*t.randn((N)), dim=-1)
    cat_pi_z = t.distributions.Categorical(pi_z)
    pi_k = t.softmax(l*t.randn((N,N)), dim=-1)

    best_loss = CrossEntropyLoss(pi_z=pi_z)(pi_k, pi_k)
    entropy.append(best_loss.unsqueeze(0))

    model = AE_bigram(N, d, pi_z, pi_k, train_U=True)
    dict = train(model, batch_size="opt", epochs=10)
    div = dict['Div'][-1]
    Div.append(div.unsqueeze(0))

    bound_1, bound_2, bound_3, bound_4, bound_5, bound_6, bound_7 = Q_bound(model, pi_k, pi_z)
    Bound_1.append((bound_1).unsqueeze(0))
    Bound_2.append((bound_2).unsqueeze(0))
    Bound_3.append((bound_3).unsqueeze(0))
    Bound_4.append((bound_4).unsqueeze(0))
    Bound_5.append((bound_5).unsqueeze(0))
    Bound_6.append((bound_6).unsqueeze(0))
    Bound_7.append((bound_7).unsqueeze(0))

entropy: t.Tensor = t.cat(entropy, dim=0)
entropy, indices = t.sort(entropy)

plt.plot(entropy, t.cat(Div, dim=0)[indices], label=f"divergence")
plt.plot(entropy, t.cat(Bound_1, dim=0)[indices], label=f"optimal")
plt.plot(entropy, t.cat(Bound_2, dim=0)[indices], label=f"pos maj")
plt.plot(entropy, t.cat(Bound_3, dim=0)[indices], label=f"neg maj")
plt.plot(entropy, t.cat(Bound_4, dim=0)[indices], label=f"pos and neg maj")
plt.plot(entropy, t.cat(Bound_5, dim=0)[indices], label=f"pos neg and coef maj")
plt.plot(entropy, t.cat(Bound_6, dim=0)[indices], label=f"opt all context")
plt.plot(entropy, t.cat(Bound_7, dim=0)[indices], label=f"worst")
plt.title("Bound comparison for different entropies")
plt.xlabel("Entropy")
plt.legend()
plt.show()
"""


"""Div = []
Q_bound = []
SVD_bound = []
SVD_bound_W = []
W_E_before = []
W_E_after = []
W_U_before = []
W_U_after = []
Th_W_U = []

step=5
for d in tqdm(range(2, N+2+1, step)):
    model = AE_bigram(N, d, train_U=False)

    W_E_before.append(is_W_E_opt(model, pi_k)/d)
    W_U_before.append(is_W_UW_U_Id(model)/d)

    dict = train(model, batch_size="opt")

    W_E_after.append(is_W_E_opt(model, pi_k)/d)
    W_U_after.append(is_W_UW_U_Id(model)/d)
    Th_W_U.append([np.sqrt(32*np.log(N+1)/d)])

    Div.append(dict["Div"][-1])
    Q_bound.append(model.Q_bound(pi_k, pi_z))
    SVD_bound.append(model.SVD_bound(pi_k, pi_z))
    SVD_bound_W.append(model.SVD_bound_W(pi_k, pi_z))

plt.plot([i for i in range(2, N+2+1, step)], Div, label="divergence")
plt.plot([i for i in range(2, N+2+1, step)], Q_bound, label="quadratic bound")
plt.plot([i for i in range(2, N+2+1, step)], SVD_bound, label="SVD bound")
#plt.plot([i for i in range(2, N+2+1, step)], SVD_bound_W, label="SVD_W bound")
plt.xlabel("Hidden dimension")
plt.ylabel("Normalized loss")
plt.title(f"Loss and its approximation for {N} tokens")
plt.legend()
plt.show()

plt.plot([i for i in range(2, N+2, step)], W_E_before[:-1], label="before training")
plt.plot([i for i in range(2, N+2, step)], W_E_after[:-1], label="after training")
plt.xlabel("Hidden dimension")
plt.ylabel("Mean L2 distance")
plt.title("Distance of the learned W_E to its optimal approximation")
plt.legend()
plt.show()

plt.plot([i for i in range(2, N+2, step)], W_U_before[:-1], label="before training")
plt.plot([i for i in range(2, N+2, step)], W_U_after[:-1], label="after training")
plt.plot([i for i in range(2, N+2, step)], Th_W_U[:-1], label="Th bound")
plt.xlabel("Hidden dimension")
plt.ylabel("Mean L2 distance")
plt.title("Distance of W_U(W_U^TW_U)^-1W_U^T to I_d")
plt.legend()
plt.show()"""