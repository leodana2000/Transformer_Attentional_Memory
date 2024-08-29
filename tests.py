"""
A file to test if the main .py works as intended.
"""

# Tests
import unittest
import torch as t
import numpy as np

#Tests for utils
from utils import power_unif_law, entropy, generate_data

class Test_power_unif_law(unittest.TestCase):
    def test_has_zeros(self):
        N = 10
        alphas = [0.2, 0.5, 1]
        nb_tokens = [10, 7, 3]

        pi = power_unif_law(alphas, nb_tokens, N)

        for pi_j, nb_token  in zip(pi, nb_tokens):
            nb_zero = (pi_j == 0.).to(t.float).sum(-1).mean().to(t.int).item()
            self.assertEqual(nb_zero, N-nb_token)

    def test_sum_to_one(self):
        N = 10
        alphas = [0.2, 0.5, 1, 0.9, 1]
        nb_tokens = [10, 7, 3, 5, 10]

        pi = power_unif_law(alphas, nb_tokens, N)

        for pi_j in pi:
            ones = pi_j.sum(-1).mean()
            self.assertAlmostEqual(ones, 1.0, delta=1e-5)


class Test_entropy(unittest.TestCase):
    def test_edge_low(self):
        N = 10
        alphas = [0.2, 0.5, 1]
        nb_tokens = [6, 7, 1]

        pi = power_unif_law(alphas, nb_tokens, N)

        ent = entropy(pi)
        self.assertAlmostEqual(ent, 0., delta=1e-5)

    def test_edge_high(self):
        N = 10
        alphas = [0.2, 0.5, 0.8, 0.9, 1]
        nb_tokens = [6, 7, 10, 10, 10]

        pi = power_unif_law(alphas, nb_tokens, N)

        ent = entropy(pi)
        self.assertAlmostEqual(ent, np.log(N), delta=1e-5)


class Test_generate_data(unittest.TestCase):
    def test_generate_one_two(self):
        N = 10
        alphas = [0.2, 0.5, 0.8, 0.2, 0.8]
        nb_tokens = [6, 7, 10, 2, 3]

        pi = power_unif_law(alphas, nb_tokens, N)
        context_window = len(pi)
        generate_data(1, 1, pi, context_window)
        generate_data(2, 1, pi, context_window)
        generate_data(1, 2, pi, context_window)
        generate_data(2, 2, pi, context_window)

    def test_generate_enough(self):
        N = 10
        alphas = [0.2, 0.5, 0.8]
        nb_tokens = [6, 7, 10]

        pi = power_unif_law(alphas, nb_tokens, N)
        batch_size = 467
        num_batch = 559
        context_window = len(pi) + 7

        dataloader = generate_data(batch_size, num_batch, pi, context_window)
        bnb, c = dataloader.dataset.tensors[0].shape
        self.assertEqual(bnb, batch_size*num_batch)
        self.assertEqual(c, context_window)


#Test models
from models import Transformer, Low_rank

class test_init_models(unittest.TestCase):
    def init_Transformer(self):
        d=10
        N=10
        nb_layers=2
        width=30
        parallel_heads=7
        d_head=12
        nb_head=3
        context_window=5
        pi = power_unif_law([0.2, 0.5, 0.8], [6, 7, 10], N)
        device='cpu'
        model = Transformer(d, N, nb_layers, width, parallel_heads, d_head, nb_head, context_window, pi, device=device)

        batch_size=10
        num_batch=1
        dataloader = generate_data(batch_size, num_batch, pi, context_window)
        for batch in dataloader:
            model(batch)

    def init_low_rank(self):
        d=10
        N=10
        context_window=5
        pi = power_unif_law([0.2, 0.5, 0.8], [6, 7, 10], N)
        device='cpu'
        model = Low_rank(d, N, context_window, pi, device=device)

        batch_size=10
        num_batch=1
        dataloader = generate_data(batch_size, num_batch, pi, context_window)
        for batch in dataloader:
            model(batch)


# Test train
from train import train

class test_training(unittest.TestCase):
    def test_training(self):
        d=10
        N=10
        nb_layers=2
        width=30
        parallel_heads=7
        d_head=12
        nb_head=3
        context_window=5
        pi = power_unif_law([0.2, 0.5, 0.8], [6, 7, 10], N)
        device='cpu'
        model = Transformer(d, N, nb_layers, width, parallel_heads, d_head, nb_head, context_window, pi, device=device)

        batch_size=100
        num_batch=100
        epochs=3
        dataloader = generate_data(batch_size, num_batch, pi, context_window)
        train(model, dataloader, epochs, next_token=True)
        train(model, dataloader, epochs, next_token=False)


if __name__ == '__main__':
    unittest.main()