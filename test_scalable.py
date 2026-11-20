"""Numerical/regression tests for categorical TT and non-enumerative MPOs."""
import unittest
from itertools import product
import numpy as np

from tensor import (amplitude_state, diagonal_mpo, tt_svd, reconstruct,
                    contract, identity_mpo, evidence, posterior,
                    conditional_scores, tt_scalar_count, tt_bond_ranks)
from scalable import (correlated_born_mps, demo_correlations,
                      sample_born_mixture, feeder_spec, terminal_model,
                      ac_terminal_voltages, ac_calibrated_terminal_models,
                      fourier_deadband_mpo, fourier_uniform_error_bound,
                      direct_deadband_scores)
from powerflow import ACTIONS, DEADBAND, LEVELS
from probability_tt import (fit_mixture_em,mixture_probability_mps,
                            probability_scores,probability_posterior,
                            probability_contract)
from benchmark import experiment


class TestCategoricalTensor(unittest.TestCase):
    def test_native_categorical_sites_and_contraction(self):
        rng = np.random.default_rng(9)
        p = rng.random((3, 3, 3, 3))
        p /= p.sum()
        energy = rng.random(p.shape)
        psi = amplitude_state(p)
        mpo = diagonal_mpo(energy)
        self.assertEqual(len(psi), 4)  # Previously eight redundant binary sites.
        self.assertEqual(tuple(c.shape[1] for c in psi), (3,)*4)
        np.testing.assert_allclose(reconstruct(psi)**2, p, atol=1e-12)
        np.testing.assert_allclose(reconstruct(mpo), energy, atol=1e-12)
        seen = {1: 2, 3: 1}
        scores, den = conditional_scores(psi, [mpo], seen)
        direct_den = p[:, 2, :, 1].sum()
        self.assertAlmostEqual(den, direct_den, places=12)
        self.assertAlmostEqual(scores[0], (p[:, 2, :, 1]*energy[:, 2, :, 1]).sum()/direct_den)
        np.testing.assert_allclose(posterior(psi, seen, 0),p[:, 2, :, 1].sum(axis=1)/direct_den)

    def test_truncated_born_distribution_remains_normalized(self):
        p = np.random.default_rng(8).random((3,)*5)
        p /= p.sum()
        psi = amplitude_state(p, max_bond=2)
        self.assertTrue(all(rank <= 2 for rank in tt_bond_ranks(psi)))
        self.assertAlmostEqual(contract(psi, identity_mpo(psi)), 1., places=11)
        self.assertGreaterEqual((reconstruct(psi)**2).min(), 0.)
        self.assertGreater(np.max(np.abs(reconstruct(psi)**2-p)), 1e-6)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            amplitude_state(p, max_bond=0)

    def test_chance_tolerance_uses_exact_safety_operator(self):
        p = np.ones((3, 3, 3)) / 27
        psi = amplitude_state(p)
        zero = diagonal_mpo(np.zeros_like(p))
        bad = np.zeros_like(p)
        bad[2, 0, 1] = 1.
        unsafe = diagonal_mpo(bad)
        scores, _, risk = conditional_scores(psi, [zero], {1: 0, 2: 1}, [unsafe],
                                             return_risks=True)
        self.assertTrue(np.isinf(scores[0]))
        self.assertAlmostEqual(risk[0], 1/3, places=12)
        scores, _, _ = conditional_scores(psi, [zero], {1: 0, 2: 1}, [unsafe],
                                          risk_tolerance=.35, return_risks=True)
        self.assertAlmostEqual(scores[0], 0)
        with self.assertRaises(ValueError):
            conditional_scores(psi, [zero], {}, [unsafe], risk_tolerance=-.01)

    def test_relative_tolerance_and_one_site(self):
        t = np.array([[1., 0., 0.], [0., 1e-4, 0.], [0., 0., 0.]])
        cores = tt_svd(t, relative_tolerance=1e-3)
        self.assertEqual(tt_bond_ranks(cores), (1,))
        self.assertLessEqual(np.linalg.norm(reconstruct(cores)-t),
                             1e-3*np.linalg.norm(t)+1e-12)
        self.assertEqual(reconstruct(tt_svd(np.array([1., 2., 3.]))).shape, (3,))


class TestAnalyticTensor(unittest.TestCase):
    def test_latent_born_matches_dense_and_correlated(self):
        p, w = demo_correlations(4)
        psi = correlated_born_mps(p,w)
        self.assertEqual(tt_bond_ranks(psi), (2,2,2))
        exact_amplitude = sum(w[k]*np.einsum('a,b,c,d->abcd',
                          *(np.sqrt(p[k,i]) for i in range(4))) for k in range(2))
        exact_amplitude /= np.linalg.norm(exact_amplitude)
        np.testing.assert_allclose(reconstruct(psi),exact_amplitude,atol=1e-12)
        density = exact_amplitude**2
        low_pair = density[0, :, :, 0].sum()
        low_0 = density[0].sum()
        low_last = density[:, :, :, 0].sum()
        self.assertGreater(low_pair,low_0*low_last)
        seen={0:1,3:1}
        states=sample_born_mixture(p,w,seen,30000,np.random.default_rng(19))
        self.assertTrue(np.all(states[:,0]==1) and np.all(states[:,3]==1))
        for i in [1,2]:
            freq = np.bincount(states[:,i],minlength=3)/len(states)
            np.testing.assert_allclose(freq,posterior(psi,seen,i),atol=.016)

    def test_analytic_fourier_operator_and_uniform_error(self):
        for n in [3,5]:
            spec=feeder_spec(n)
            c,f=terminal_model(n,ACTIONS[4],spec)
            domain=1.1*max(abs(c+f.min(axis=1).sum()),abs(c+f.max(axis=1).sum()))
            harmonics=24
            op=fourier_deadband_mpo(c,f,DEADBAND,domain,harmonics)
            self.assertEqual(len(op),n)
            self.assertEqual(max(tt_bond_ranks(op)),1+2*harmonics)
            all_states=np.array(list(product(range(3),repeat=n)))
            expected=direct_deadband_scores(all_states,[ACTIONS[4]],spec,DEADBAND)[:,0]
            np.testing.assert_array_less(np.abs(reconstruct(op).ravel()-expected),
                                         fourier_uniform_error_bound(domain,harmonics)+1e-12)
            with self.assertRaisesRegex(ValueError,'domain'):
                fourier_deadband_mpo(c,f,DEADBAND,domain*.1,harmonics)

    def test_calibrated_voltage_affine_matches_reference_and_reduces_bias(self):
        spec=feeder_spec(8)
        models,calls=ac_calibrated_terminal_models(8,ACTIONS,spec)
        self.assertEqual(calls,len(ACTIONS)*(2*8+1))
        ref=np.full(8,.1)
        for a,(c,f) in zip(ACTIONS,models):
            target=ac_terminal_voltages(ref,a,spec)[-1]
            self.assertAlmostEqual(1+c+f[:,1].sum(),target,places=11)

    def test_scaling_does_not_enumerate_dense_high_dimensional_space(self):
        row=experiment(10,harmonics=8,samples=2000,ac_check=20)
        self.assertEqual(row['dense_states'],59049)
        self.assertEqual(row['probability_mps_max_rank'],2)
        self.assertEqual(row['operator_mpo_max_rank'],17)
        self.assertLess(row['operator_mpo_scalars_per_action'],row['dense_operator_scalars_per_action'])
        self.assertEqual(row['offline_ac_calls'],len(ACTIONS)*(2*10+1))
        self.assertGreaterEqual(row['ac_terminal_voltage_mae'],0)
        self.assertIsNone(row['max_error_vs_exact_enumeration'])

    def test_learned_mixture_probability_mps_and_nonnegative_contraction(self):
        p,w=demo_correlations(5)
        train=sample_born_mixture(p,w,{},4000,np.random.default_rng(9))
        fitted=fit_mixture_em(train,components=3,iterations=100,restarts=2,seed=99)
        prob=mixture_probability_mps(fitted['weights'],fitted['local_probabilities'])
        self.assertTrue(np.all(reconstruct(prob)>=0))
        self.assertAlmostEqual(reconstruct(prob).sum(),1,places=11)
        seen={0:1,4:1}
        op=diagonal_mpo(np.random.default_rng(7).random((3,)*5))
        scores,den=probability_scores(prob,(op,),seen)
        dense=reconstruct(prob)
        dval=dense[1,:,:, :,1]
        self.assertAlmostEqual(den,dval.sum(),places=11)
        self.assertAlmostEqual(scores[0],(dval*reconstruct(op)[1,:,:,:,1]).sum()/dval.sum(),places=11)
        np.testing.assert_allclose(probability_posterior(prob,seen,2),
                                   dval.sum(axis=(0,2))/dval.sum(),atol=1e-11)

    def test_learned_high_dimensional_mode(self):
        row=experiment(10,harmonics=8,samples=2000,ac_check=10,
                       probability_source='em-learned',train_samples=2000,
                       components=3)
        self.assertGreater(row['model_training_seconds'],0)
        self.assertEqual(row['train_samples'],2000)
        self.assertGreater(row['max_score_error_vs_true_mps'],0)

    def test_small_scale_exact_enumeration_checks_surrogate(self):
        row=experiment(4,harmonics=16,samples=1000,ac_check=0)
        self.assertLessEqual(row['max_error_vs_exact_enumeration'],
                             row['operator_uniform_error_bound'])
        self.assertIsNone(row['ac_regret_vs_best_sample'])
        self.assertGreater(row['evidence_probability'],0)


if __name__ == '__main__':
    unittest.main()
