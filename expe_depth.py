import torch as t
t.set_num_threads(8)
import pandas as pd
from train import train
from models import Transformer, AoT
from utils import generate_data, power_unif_law
from tqdm import tqdm

folder_name = "Scaling laws"
device = "cpu"



""" Experiment 1. Scaling laws on H with fixed d=d_head. """
t.manual_seed(2222)

# Model parameters.
N = 50
nb_layers = 5 # Depth of the network
nb_head = 1
n_gram = 3
context_window = n_gram

# Distribution parameters.
alphas = [1, 1, 1.]
nb_tokens=[100, 100, 1]
pi = power_unif_law(alphas, nb_tokens, N)

# Training parameters.
batch_size=2**10
num_batch=1000
lr=5e-4
epochs=10
repetition = 2
Data = generate_data(batch_size=batch_size, num_batch=num_batch, pi=pi, context_window=context_window)

# Scaling parameters
d = 10

d_head=d 

mean_accuracy = []
para_list = []
N_list = []
d_list = []
d_head_list = []

for para in tqdm([1, 6, 11, 16, 21]):
    accuracy = 0.

    for _ in range(repetition):
        model = AoT(d, N, nb_layers, para, d_head, nb_head, context_window, pi, device=device)
        model.to(device)

        dict = train(model, Data, epochs, lr=lr, next_token=True)
        acc = sum(dict['Acc'][-101:-1])/100
            
        accuracy += acc

    mean_accuracy.append(accuracy/repetition)
    N_list.append(N)
    d_list.append(d)
    d_head_list.append(d_head)
    para_list.append(para)

results = {
    'acc': mean_accuracy,
    'para': para_list,
    'N': N_list,
    'd': d_list,
    'd_head': d_head_list,
}

# We save the results as a dataframe.
data = pd.DataFrame(results)
data.to_csv(f'{folder_name}/Data_exp_1_{7}_depth.csv', index=False)

""" Experiment 2. Scaling laws on d_head, with d!=d_head and H (=para) fixed. """
t.manual_seed(2222)

# Model parameters.
N = 50
d = 10
para = 20
nb_layers = 5 # Depth of the network
nb_head = 1
n_gram = 3
context_window = n_gram

# Distribution parameters.
alphas = [1, 1, 1]
nb_tokens=[100, 100, 1]
pi = power_unif_law(alphas, nb_tokens, N)

# Training parameters.
batch_size=2**10
num_batch=1000
lr=5e-4
epochs=10
repetition = 2
Data = generate_data(batch_size=batch_size, num_batch=num_batch, pi=pi, context_window=context_window)

# Scaling parameters

mean_accuracy = []
para_list = []
N_list = []
d_list = []
d_head_list = []
for d_head in tqdm([1, 3, 5, 7, 10]):
    accuracy = 0

    for _ in range(repetition):
        model = AoT(d, N, nb_layers, para, d_head, nb_head, context_window, pi, device=device)
        model.to(device)

        dict = train(model, Data, epochs, lr=lr, next_token=True)
        acc = sum(dict['Acc'][-101:-1])/100
        
        accuracy += acc

    mean_accuracy.append(accuracy/repetition)
    N_list.append(N)
    d_list.append(d)
    d_head_list.append(d_head)
    para_list.append(para)

results = {
    'acc': mean_accuracy,
    'para': para_list,
    'N': N_list,
    'd': d_list,
    'd_head': d_head_list,
}

# We save the results as a dataframe.
data = pd.DataFrame(results)
data.to_csv(f'{folder_name}/Data_exp_2_depth.csv', index=False)

""" Experiment 5. Scaling laws on the width of Transformer using MLPs. """
t.manual_seed(3333)

# Model parameters.
N = 50
para = 1
nb_layers = 5 # Depth of the network
nb_head = 1
n_gram = 3
context_window = n_gram

# Distribution parameters.
alphas = [1, 1, 1]
nb_tokens=[100, 100, 1]
pi = power_unif_law(alphas, nb_tokens, N)

# Training parameters.
batch_size=2**10
num_batch=1000
lr=5e-4
epochs=10
repetition = 2
Data = generate_data(batch_size=batch_size, num_batch=num_batch, pi=pi, context_window=context_window)

for d, exp_num in zip([7, 10, 13], [4, 7, 10]):
    d_head = d
    min_width = 2*d*(1-1)
    max_width = 2*d*(21-1)
    step = 2*d*5

    mean_accuracy = []
    para_list = []
    N_list = []
    d_list = []
    d_head_list = []
    width_list = []
    for width in tqdm(range(min_width, max_width+1, step)):
        accuracy = 0

        for _ in range(repetition):
            model = Transformer(d, N, nb_layers, width, para, d_head, nb_head, context_window, pi, device=device)
            model.to(device)

            dict = train(model, Data, epochs, lr=lr, next_token=True)
            acc = sum(dict['Acc'][-101:-1])/100
            
            accuracy += acc
            print(accuracy)

        mean_accuracy.append(accuracy/repetition)
        N_list.append(N)
        d_list.append(d)
        d_head_list.append(d_head)
        para_list.append(para)
        width_list.append(width)

    results = {
        'acc': mean_accuracy,
        'para': para_list,
        'N': N_list,
        'd': d_list,
        'd_head': d_head_list,
        'width': width_list,
    }

    # We save the results as a dataframe.
    data = pd.DataFrame(results)
    data.to_csv(f'Scaling laws/Data_exp_5_{exp_num}_depth.csv', index=False)


for d, exp_num in zip([7, 13], [4, 10]):
    d_head = d
    min_para = 1
    max_para = 21
    step = 5

    mean_accuracy = []
    para_list = []
    N_list = []
    d_list = []
    d_head_list = []
    width_list = []
    for width in tqdm(range(min_para, max_para+1, step)):
        accuracy = 0

        for _ in range(repetition):
            model = AoT(d, N, nb_layers, para, d_head, nb_head, context_window, pi, device=device)
            model.to(device)

            dict = train(model, Data, epochs, lr=lr, next_token=True)
            acc = sum(dict['Acc'][-101:-1])/100
            
            accuracy += acc
            print(accuracy)

        mean_accuracy.append(accuracy/repetition)
        N_list.append(N)
        d_list.append(d)
        d_head_list.append(d_head)
        para_list.append(para)
        width_list.append(width)

    results = {
        'acc': mean_accuracy,
        'para': para_list,
        'N': N_list,
        'd': d_list,
        'd_head': d_head_list,
        'width': width_list,
    }

    # We save the results as a dataframe.
    data = pd.DataFrame(results)
    data.to_csv(f'Scaling laws/Data_exp_1_{exp_num}_depth.csv', index=False)