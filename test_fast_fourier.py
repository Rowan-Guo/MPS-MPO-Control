import unittest

import numpy as np

from fast_fourier import FactorizedFourier
from probability_tt import mixture_probability_mps, probability_scores, probability_posterior
from scalable import fourier_deadband_mpo


class TestFactorizedFourier(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(153)
        self.constants = rng.uniform(-.1, .1, size=(3, 2))
        self.contributions = rng.uniform(-.01, .02, size=(3, 2, 5, 3))
        self.weights = np.array([.2, .1, .7])
        self.local = rng.dirichlet(np.ones(3), size=(3, 5))
        self.prob = mixture_probability_mps(self.weights, self.local)
        self.fast = FactorizedFourier(self.constants, self.contributions, .02, 8, domain=.4)

    def dense(self, constants, obs):
        ops = [fourier_deadband_mpo(constants[a, m], self.contributions[a, m], .02, .4, 8)
               for a, m in np.ndindex(constants.shape)]
        scores, den = probability_scores(self.prob, ops, obs)
        return scores.reshape(constants.shape), den

    def test_missing_partial_and_full_observation(self):
        for obs in ({}, {0: 1, 3: 0}, {i: i % 3 for i in range(5)}):
            with self.subTest(observed=obs):
                expected, den = self.dense(self.constants, obs)
                scores, margins, mass = self.fast.scores_and_marginals(self.weights, self.local, obs)
                np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-13)
                self.assertAlmostEqual(den, mass)
                for site, row in enumerate(margins):
                    reference = (np.eye(3)[obs[site]] if site in obs else
                                 probability_posterior(self.prob, obs, site))
                    np.testing.assert_allclose(row, reference, rtol=0, atol=1e-13)

    def test_shifted_phases_reuse_and_domain_enforcement(self):
        obs = {2: 1}
        shifts = np.stack([self.constants * 0, self.constants * .25, self.constants * -.5])
        evaluate = self.fast.shift_evaluator(self.weights, self.local, obs)
        actual = evaluate(shifts)
        for row, shift in zip(actual, shifts, strict=True):
            expected, _ = self.dense(self.constants + shift, obs)
            np.testing.assert_allclose(row, expected, rtol=0, atol=1e-13)
        with self.assertRaisesRegex(ValueError, 'shifted voltage'):
            evaluate(shifts + 1.)
        with self.assertRaisesRegex(ValueError, 'shape|dimensions|array'):
            evaluate(np.zeros(3))

    def test_invalid_evidence(self):
        locals_zero = self.local.copy()
        locals_zero[:, 0, :] = [0., 0., 1.]
        with self.assertRaisesRegex(ValueError, 'zero probability'):
            self.fast.scores(self.weights, locals_zero, {0: 1})
        with self.assertRaisesRegex(ValueError, 'invalid'):
            self.fast.scores(self.weights, self.local, {6: 1})
