import torch as t
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Measuring the Recall of a random matrix of size N and rank d.
"""

#%% Recall with d=log(N)

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N_max = 400
start = 2
step = 1
repet = 30

L = []
for N in tqdm(range(start, N_max, step)):
    acc = 0.
    d = N//2
    for _ in range(repet):
        P = t.randn((N, d))/np.sqrt(d)
        W = t.zeros((d,d))
        for i in range(1, N):
            W += P[i].unsqueeze(-1)@P[i-1].unsqueeze(0)
        PP = P@W@P.T
        ind = t.argmax(PP, dim=-1)
        acc += (ind[1:] == t.arange(N)[:-1]).to(t.float).mean().item()
    L.append(acc/repet)

plt.plot([i for i in range(start, N_max, step)], L)
plt.xlabel("N = 2d")
plt.ylabel("Recall")
plt.title("Recall of random vectors with N=2d")
plt.show()

#%% Recall with d=log(N)

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N_max = 600
start = 2
step = 3
repet = 30

L = []
for N in tqdm(range(start, N_max, step)):
    acc = 0.
    d = max(int(4*np.log(N)), 2)
    for _ in range(repet):
        P = t.randn((N, d))/np.sqrt(d)
        W = t.zeros((d,d))
        for i in range(1, N):
            W += P[i].unsqueeze(-1)@P[i-1].unsqueeze(0)
        PP = P@W@P.T
        ind = t.argmax(PP, dim=-1)
        acc += (ind[1:] == t.arange(N)[:-1]).to(t.float).mean().item()
    L.append(acc/repet)

plt.plot([i for i in range(start, N_max, step)], L)
plt.xlabel("N = e^(d/4)")
plt.ylabel("Recall")
plt.title("Recall of random vectors with d=4log(N)")
plt.show()

#%% Recall as N grow
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

d = 100
N_max = 20*d
start = 2
step = 10
repet = 30

L = []
for N in tqdm(range(start, N_max, step)):
    acc = 0.
    for _ in range(repet):
        P = t.randn((N, d))/np.sqrt(d)
        W = t.zeros((d,d))
        for i in range(1, N):
            W += P[i].unsqueeze(-1)@P[i-1].unsqueeze(0)
        PP = P@W@P.T
        ind = t.argmax(PP, dim=-1)
        acc += (ind[1:] == t.arange(N)[:-1]).to(t.float).mean().item()  
    L.append(acc/repet)

plt.plot([i for i in range(start, N_max, step)], L)
if d >= start:
    plt.scatter([d+start], [L[(start+d)//step]], label=f"d={d}")
    plt.legend()
plt.xlabel("N")
plt.ylabel("Recall")
plt.title(f"Recall of random vectors with d={d}")
plt.show()

#%% Recal as d grow
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N = 400
N_max = 200
step = 5
start = 2
repet = 30

L = []
for d in tqdm(range(start, N_max, step)):
    acc = 0.
    for _ in range(repet):
        P = t.randn((N, d))/np.sqrt(d)
        W = t.zeros((d,d))
        for i in range(1, N):
            W += P[i].unsqueeze(-1)@P[i-1].unsqueeze(0)
        PP = P@W@P.T
        ind = t.argmax(PP, dim=-1)
        acc += (ind[1:] == t.arange(N)[:-1]).to(t.float).mean().item()  
    L.append(acc/repet)

plt.plot([i for i in range(start, N_max, step)], L)
if N <= N_max:
    plt.scatter([N+start], [L[(start+N)//step]], label=f"N={N}")
    plt.legend()
    plt.xlabel("d")
else:
    plt.xlabel(f"d, N={N}")
plt.ylabel("Recall")
plt.title(f"Recall of random vectors with N={N}")
plt.show()

#%% argmax as N grow

d = 10
N_max = 100*d
step = d//2
repet = 20

L = []
for N in tqdm(range(d, N_max, step)):
    acc = 0.
    for _ in range(repet):
        P = t.randn((N, d))/np.sqrt(d)
        PP = P@P.T
        ind = t.argmax(PP, dim=-1)
        acc += (ind == t.arange(N)).to(t.float).mean().item()
    L.append(acc/repet)

plt.plot([i for i in range(d, N_max, step)], L)
if d >= start:
    plt.scatter([d+start], [L[(start+d)//step]], label=f"d={d}")
    plt.legend()
plt.xlabel("N")
plt.ylabel("Accuracy")
plt.show()

#%% argmax as d grow
N = 1000
N_max = 50
step = 1
repet = 20

L = []
for d in tqdm(range(1, N_max, step)):
    acc = 0.
    for _ in range(repet):
        P = t.randn((N, d))/np.sqrt(d)
        PP = P@P.T
        ind = t.argmax(PP, dim=-1)
        acc += (ind == t.arange(N)).to(t.float).mean().item()
    L.append(acc/repet)

plt.plot([i for i in range(1, N_max, step)], L)
if N <= N_max:
    plt.scatter([N+start], [L[(start+N)//step]], label=f"N={N}")
    plt.legend()
    plt.xlabel("d")
else:
    plt.xlabel(f"d, N={N}")
plt.ylabel("Accuracy")
plt.show()
#%%

