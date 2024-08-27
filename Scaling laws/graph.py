"""
A file that plots the scaling laws in the above folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Exp 1: We measure the accuracy for H heads and d_head = d, meaning that the embedding dimension is also the head dimension.
We find that the scaling in H is linear, and cubic in d.
"""

do_plot = False

A = []
for i in range(10+1):
    data_1: pd.DataFrame = pd.read_csv(f'Scaling laws/exp_{i}.csv')
    para_list = data_1['para'].to_list()
    accuracy = data_1['acc'].to_list()
    N = data_1['N'].to_list()[0]
    d = data_1['d'].to_list()[0]
    if do_plot:
        plt.plot(para_list, accuracy, label="AoT")
        plt.plot(para_list, [1/N for _ in range(len(para_list))], color='black', label="Random")
        plt.plot(para_list, [(1-1/N)*para*d/(N**2)+1/N for para in para_list], label="Our lower bound")
        plt.plot(para_list, [(1-1/N)*(para*(d-1)+1)/(N**2)+1/N for para in para_list], label="Previous lower bound")
        plt.legend()
        plt.xlabel("Number of parallel head")
        plt.ylabel("Proportion of remembered sentences")
        plt.title(f"Scaling laws for N=50, d={d}")

    # We filter out if the last points are constant at accuracy 1
    for i, acc in enumerate(accuracy):
        if acc > 0.95:
            accuracy = accuracy[:i+1]
            para_list = para_list[:i+1]
            break

    y = sum(accuracy)/len(accuracy)
    x = sum(para_list)/len(para_list)
    Y = np.array(accuracy) - y
    X = np.array(para_list) - x
    a = np.sum(Y*X)/np.sum(X*X)
    A.append(a*(N**2))
    b = y-a*x
    Z = np.array(para_list)*a+b

    # We plot the linear regression
    if do_plot:
        plt.plot(para_list, Z, color= "C0", label="Linear Approx")
        plt.show()

D = np.array([d for d in range(3, 13+1)])

plt.plot(D, A, label="Empirical coefficient")

y = sum(np.log(A))/len(A)
x = sum(np.log(D))/len(D)
a = 2
b = y - a*x
Z = np.exp(a*np.log(D)+b)
plt.plot(D, Z, label="Quadratic approx")
print(f"a:{a}, c:{np.exp(b)}")

a = 3
b = y - a*x
Z = np.exp(a*np.log(D)+b)
plt.plot(D, Z, label="Cubic approx")
print(f"a:{a}, c:{np.exp(b)}")

Y = np.log(A) - y
X = np.log(D) - x
a = np.sum(Y*X)/np.sum(X*X)
b = y - a*x
Z = np.exp(a*np.log(D)+b)
plt.plot(D, Z, label="Monomial approx")
print(f"a:{a}, c:{np.exp(b)}")

y = sum(np.log(A))/len(A)
x = sum(D)/len(D)
Y = np.log(A) - y
X = D - x
a = np.sum(Y*X)/np.sum(X*X)
b = y - a*x
Z = np.exp(a*D+b)

plt.plot(D, Z, label="Exp approx")
print(f"a:{a}, c:{np.exp(b)}")


plt.xlabel("Embedding dimension")
plt.ylabel("Scaling law coefficient")
plt.title(f"Scaling laws in d")

plt.legend()
plt.show()


"""
Exp 2: We measure the accuracy for H heads and d_head != d, meaning that the embedding dimension is not the head dimension. 
In particular, the head dimension is varying while the embedding dimensio satys the same.
We find that the scaling in H is linear, and ???.
"""


do_plot = False

A = []
for i in range(8+1):
    data_1 = pd.read_csv(f'Scaling laws/exp_{i}_2.csv')
    para_list = data_1['para'].to_list()
    accuracy = data_1['acc'].to_list()
    N = data_1['N'].to_list()[0]
    d = data_1['d'].to_list()[0]
    d_head = data_1['d_head'].to_list()[0]
    if do_plot:
        plt.plot(para_list, accuracy, label="AoT")
        plt.plot(para_list, [1/N for _ in range(len(para_list))], color='black', label="Random")
        plt.plot(para_list, [(1-1/N)*para*d/(N**2)+1/N for para in para_list], label="Our lower bound")
        plt.plot(para_list, [(1-1/N)*(para*(d-1)+1)/(N**2)+1/N for para in para_list], label="Previous lower bound")
        plt.legend()
        plt.xlabel("Number of parallel head")
        plt.ylabel("Proportion of remembered sentences")
        plt.title(f"Scaling laws for N=50, d=10, d_head={d_head}")

    # We filter out if the last points are constant at accuracy 1
    for i, acc in enumerate(accuracy):
        if acc > 0.95:
            accuracy = accuracy[:i+1]
            para_list = para_list[:i+1]
            break

    y = sum(accuracy)/len(accuracy)
    x = sum(para_list)/len(para_list)
    Y = np.array(accuracy) - y
    X = np.array(para_list) - x
    a = np.sum(Y*X)/np.sum(X*X)
    A.append(a*(N**2))
    b = y-a*x
    Z = np.array(para_list)*a+b

    # We plot the linear regression
    if do_plot:
        plt.plot(para_list, Z, color= "C0", label="Linear Approx")
        plt.show()

A = A[1:]
D = np.array([d for d in range(3+1, 8+1)]+[10, 12, 15])

plt.plot(D, A, label="Empirical coefficient")

y = sum(np.log(A))/len(A)
x = sum(np.log(D))/len(D)

a = 1
b = y - a*x
Z = np.exp(a*np.log(D)+b)
c = sum(A)/len(A) - sum(Z)/len(Z)
plt.plot(D, Z+c, label="Linear approx")
print(f"a:{a}, c:{np.exp(b)}, d:{c}")

a = 2
b = y - a*x
Z = np.exp(a*np.log(D)+b)
plt.plot(D, Z, label="Quadratic approx")
print(f"a:{a}, c:{np.exp(b)}")

a = 3
b = y - a*x
Z = np.exp(a*np.log(D)+b)
plt.plot(D, Z, label="Cubic approx")
print(f"a:{a}, c:{np.exp(b)}")

Y = np.log(A) - y
X = np.log(D) - x
a = np.sum(Y*X)/np.sum(X*X)
b = y - a*x
Z = np.exp(a*np.log(D)+b)
plt.plot(D, Z, label="Monomial approx")
print(f"a:{a}, c:{np.exp(b)}")

x = sum(D)/len(D)
X = D - x
a = np.sum(Y*X)/np.sum(X*X)
b = y - a*x
Z = np.exp(a*D+b)
c = sum(A)/len(A) - sum(Z)/len(Z)

plt.plot(D, Z+c, label="Exp approx")
print(f"a:{a}, c:{np.exp(b)}")


plt.xlabel("Embedding dimension")
plt.ylabel("Scaling law coefficient")
plt.title(f"Scaling laws in d")

plt.legend()
plt.show()