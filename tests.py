
# Tests
import unittest
import torch as t
import numpy as np

if __name__ == '__main__':
    unittest.main()

#Tests for utils
from utils import power_unif_law, entropy, generate_data, layer_norm

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

        dataloader = generate_data(batch_size, num_batch, pi, context_window, one_extra=True)
        _, c = dataloader.dataset.tensors[0].shape
        self.assertEqual(c, context_window+1)


class Test_layer_norm(unittest.TestCase):
    def test_zero(self):
        x = t.zeros((5, 5))
        y = layer_norm(x)
        self.assertAlmostEquals(t.abs(x-y).sum(), 0., delta=1e-5)

    def test_norm(self):
        x = t.randn((6, 3, 9))
        y = layer_norm(x)
        self.assertAlmostEquals(t.abs((y**2).mean(-1) - 1).sum(), 0., delta=1e-5)


#Test train
#TODO

#Test interp
#TODO