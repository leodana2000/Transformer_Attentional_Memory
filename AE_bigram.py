import numpy as np
import torch as t
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
We test here if the empirical loss and the order 2 approximations are close.
"""

#meta parameters
N = 100
d = 20

#bigram distributions
lamb_z = 1
pi_z = t.softmax(lamb_z*t.randn((N)), dim=-1)
cat_pi_z = t.distributions.Categorical(pi_z)

lamb_k = 2
pi_k = t.softmax(lamb_k*t.randn((N,N)), dim=-1)


unif = t.softmax(t.zeros_like(pi_z), dim=-1)
def CrossEntropyLoss(pi_z=unif):
    def loss(pi_k, pi_p):
        best_loss = -t.einsum('b, bN, bN -> ', pi_z, pi_k, t.log(pi_p))
        return best_loss
    return loss


best_loss = CrossEntropyLoss(pi_z=pi_z)(pi_k, pi_k)
print(f"The best achievable loss is {best_loss}.")


class AE_bigram(t.nn.Module):
    def __init__(self, N, d, train_U=True):
        super().__init__()
        self.W_E = t.nn.Linear(d, N, bias=False)
        self.W_U = t.nn.Linear(d, N, bias=False)
        self.W_U.requires_grad_ = train_U

        self.N = N
        self.d = d

    def forward(self, x):
        return t.softmax(self.W_U(self.W_E.weight[x]), dim=-1)
    
    def loss_approx(self, pi_k, pi_z):
        x = t.arange(0, self.N, 1)
        with t.no_grad():
            logits = self.W_U(self.W_E.weight[x])
            Z = logits - t.log(pi_k)
            var_Z =  (pi_k*(Z**2)).sum(-1) - ((pi_k*Z).sum(-1))**2
            return (var_Z*pi_z).sum(0)/2


def train(model, lr=1e-4, epochs=15, batch_size: int | str = 2**10, num_batch=1000, compute_approx=False):
    if batch_size == "opt":
        f_loss = CrossEntropyLoss(pi_z=pi_z)
    else:
        f_loss = CrossEntropyLoss()

    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    Loss = []
    Loss_approx = []
    for _ in tqdm(range(epochs)):
        loss_avg = 0
        loss_approx_avg = 0
        count = 0

        for b in range(num_batch):
            if batch_size == "opt":
                X = t.arange(0, model.N, 1)
            else:
                X = cat_pi_z.sample((batch_size, 1)).squeeze()

            model_pi = model(X)
            true_pi = pi_k[X]

            loss = f_loss(true_pi, model_pi)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if (num_batch - b) <= 100:
                count += 1
                loss_avg += loss.item()
                if compute_approx:
                    loss_approx_avg += model.loss_approx(pi_k, pi_z).item()

        Loss.append((loss_avg/count-best_loss)/best_loss)
        Loss_approx.append((loss_approx_avg/count)/best_loss)
    
    return {'Loss': Loss, 'Loss_approx': Loss_approx}


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


def is_mean_0(model):
    W_U = model.W_U.weight
    return t.norm(W_U.sum(-1)).item()


def is_W_UW_U_Id(model):
    W_U = model.W_U.weight
    pseudo_inv = W_U@t.inverse(W_U.T@W_U)@W_U.T
    return t.norm(pseudo_inv-t.eye(model.N)).item()


"""model1 = AE_bigram(N, d, train_U=False)
dict1 = train(model1, batch_size="opt", compute_approx=True, epochs=20)

model2 = AE_bigram(N, d, train_U=True)
dict2 = train(model2, batch_size="opt", compute_approx=True, epochs=20)

plt.plot(dict1['Loss'], label="loss for W_E")
plt.plot(dict1['Loss_approx'], label="loss approx for W_E")
plt.plot(dict2['Loss'], label="loss for W_E and W_U")
plt.plot(dict2['Loss_approx'], label="loss approx for W_E and W_U")
plt.title("Learning comparison")
plt.xlabel("epochs")
plt.ylabel("normalized loss")
plt.legend()
plt.show()"""



Loss = []
Loss_approx = []
W_E_before = []
W_E_after = []
W_U_before = [[],[]]
W_U_after = [[],[]]

step=5
for d in tqdm(range(2, N+2+1, step)):
    model = AE_bigram(N, d, train_U=False)
    W_E_before.append(is_W_E_opt(model, pi_k)/d)
    W_U_before[0].append(is_mean_0(model)/d)
    W_U_before[1].append(is_W_UW_U_Id(model)/d)
    dict = train(model, batch_size="opt", compute_approx=True)
    W_E_after.append(is_W_E_opt(model, pi_k)/d)
    W_U_after[0].append(is_mean_0(model)/d)
    W_U_after[1].append(is_W_UW_U_Id(model)/d)
    Loss.append(dict["Loss"][-1])
    Loss_approx.append(dict["Loss_approx"][-1])

"""plt.plot([i for i in range(2, N+2+1, step)], Loss, label="loss")
plt.plot([i for i in range(2, N+2+1, step)], Loss_approx, label="loss approx")
plt.xlabel("Hidden dimension")
plt.ylabel("Normalized loss")
plt.title(f"Loss and its approximation for {N} tokens")
plt.legend()
plt.show()"""

"""plt.plot([i for i in range(2, N+2, step)], W_E_before[:-1], label="before training")
plt.plot([i for i in range(2, N+2, step)], W_E_after[:-1], label="after training")
plt.xlabel("Hidden dimension")
plt.ylabel("Mean L2 distance")
plt.title("Distance of the learned W_E to its optimal approximation")
plt.legend()
plt.show()"""

"""plt.plot([i for i in range(2, N+2, step)], W_U_before[0][:-1], label="before training")
plt.plot([i for i in range(2, N+2, step)], W_U_after[0][:-1], label="after training")
plt.xlabel("Hidden dimension")
plt.ylabel("Norm")
plt.title("Mean of the learned w_U(k)")
plt.legend()
plt.show()"""

plt.plot([i for i in range(2, N+2, step)], W_U_before[1][:-1], label="before training")
plt.plot([i for i in range(2, N+2, step)], W_U_after[1][:-1], label="after training")
plt.xlabel("Hidden dimension")
plt.ylabel("Mean L2 distance")
plt.title("Distance of W_U(W_U^TW_U)^-1W_U^T to I_d")
plt.legend()
plt.show()