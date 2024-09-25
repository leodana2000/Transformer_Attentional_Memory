"""
Exp 1: We want to validate that the scaling law is linear in H, so for different values of d=d_head, we measure the accuracy as H grows. 
In this setup we use:
* N=50
* d=d_head ranging from 3 to 13
We find that before saturation, the accuracy is linear in H.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plot_all = True

Scaling_coef = []
for i in range(10+1):
    data: pd.DataFrame = pd.read_csv(f'Scaling laws/Data_exp_1_{i}.csv')
    H_list = data['para'].to_list()
    accuracy = data['acc'].to_list()
    N = data['N'].to_list()[0]
    d = data['d'].to_list()[0]

    # We filter out if the last points are constant at accuracy 1 to compute the linear scaling before it saturates.
    for i, acc in enumerate(accuracy):
        if (acc > 0.95) or (i == len(accuracy)-1):
            linear_accuracy = accuracy[:i+1]
            H_list = H_list[:i+1]
            break

    # We manually compute the linear regression.
    y = sum(linear_accuracy)/len(linear_accuracy)
    x = sum(H_list)/len(H_list)
    Y = np.array(linear_accuracy) - y
    X = np.array(H_list) - x
    a = np.sum(Y*X)/np.sum(X*X)
    Scaling_coef.append(a*(N**2))
    b = y-a*x
    Z = np.array(H_list)*a+b

    # We plot the linear regression
    if plot_all:
        plt.plot(H_list, linear_accuracy, label="AoT")
        plt.plot(H_list, [1/N for _ in range(len(H_list))], color='black', label="Random")
        plt.plot(H_list, [(1-1/N)*(para*d+d)/(N**2)+1/N for para in H_list], label="Our lower bound")
        plt.plot(H_list, [(1-1/N)*(para*(d-1)+1)/(N**2)+1/N for para in H_list], label="Previous lower bound")
        plt.plot(H_list, Z, color= "C0", linestyle='--', label='Linear regression')
        plt.xlabel("Number of Heads")
        plt.ylabel("Accuracy")
        plt.title(f"Scaling law for N={N}, d=d_head={d}.")
        plt.legend()
        plt.show()


"""
Exp 4: We validate experiment 2 and 3 by measuring the scaling laws with d=d_head. Using the scaling laws from Exp_1, 
we compute the linear scaling before saturation, and use it for a scaling laws of its own.
In this setup we use:
* N=50
* d=d_head
We find a cubic scaling for the coefficient, meaning that the accuracy has the scaling C*H*d**3.
"""

d_list = np.array([d for d in range(3, 13+1)])
log_d_list = np.log(d_list)
log_Scaling_coef = np.log(Scaling_coef)

plt.plot(d_list, Scaling_coef, label="Empirical coefficient")

y = sum(log_Scaling_coef)/len(Scaling_coef)
x = sum(log_d_list)/len(d_list)
a = 2
b = y - a*x
Z = np.exp(a*log_d_list+b)
plt.plot(d_list, Z, label="Quadratic approx")
print(f"Quadratic regression. a:{a}, c:{np.exp(b)}")

a = 3
b = y - a*x
Z = np.exp(a*log_d_list+b)
plt.plot(d_list, Z, label="Cubic approx")
print(f"Cubic regression. a:{a}, c:{np.exp(b)}")

"""Y = log_Scaling_coef - y
X = log_d_list - x
a = np.sum(Y*X)/np.sum(X*X)
b = y - a*x
Z = np.exp(a*log_d_list+b)
plt.plot(d_list, Z, label="Monomial approx")
print(f"Monomial regression. a:{a}, c:{np.exp(b)}")"""

y = sum(log_Scaling_coef)/len(Scaling_coef)
x = sum(d_list)/len(d_list)
Y = log_Scaling_coef - y
X = d_list - x
a = np.sum(Y*X)/np.sum(X*X)
b = y - a*x
Z = np.exp(a*d_list+b)

plt.plot(d_list, Z, label="Exp approx")
print(f"Exponential regression. a:{a}, c:{np.exp(b)}")

plt.xlabel("Embedding dimension")
plt.ylabel("Scaling law coefficient")
plt.title(f"Scaling laws in d")

plt.legend()
plt.show()