import torch as t

def KL_div(pi, softmax):
    return (pi*(t.log(pi)-t.log(softmax))).sum()

def soft_linear(pi, logits):
    z = logits - t.log(pi)
    Z = z - (z*pi).sum()
    return t.log((pi*t.exp(Z)).sum()), t.log(1+(pi*(t.exp(Z)-1)).sum())

def Q_bound(pi, logits):
    Z = logits-t.log(pi)
    Z = Z - (Z*pi).sum()
    Z -= t.max(Z)

    pos = (Z > 0).to(t.float)
    neg = pos-1

    return ((Z**2)*pos/8 + pi*Z*neg).sum()

for l in [0.1, 0.5, 1, 2, 3, 5]:
    N=100
    logits = l*t.randn(N)
    pi = t.softmax(logits+t.randn_like(logits)*0.1, dim=0)
    softmax = t.softmax(logits, dim=0)

    print(KL_div(pi, softmax), Q_bound(pi, logits))