"""Reproducible scaling test: correlated Born MPS + analytic linearized loss MPO.

No dense 3**sites tensor is ever formed except for OPTIONAL small-site verification.
All online scores are evaluated through MPS/MPO contractions. Monte Carlo
samples are drawn from the same known-law Born distribution for diagnostics.
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

from powerflow import ACTIONS, DEADBAND, LEVELS, loss
from tensor import (conditional_scores, posterior, reconstruct,
                    tt_bond_ranks, tt_scalar_count)
from probability_tt import (fit_mixture_em,mixture_probability_mps,
                            probability_scores,probability_posterior)
from scalable import (correlated_born_mps, demo_correlations,
                      sample_born_mixture, feeder_spec, terminal_model,
                      fourier_deadband_mpo, fourier_uniform_error_bound,
                      direct_deadband_scores, full_ac, ac_calibrated_terminal_models,
                      ac_terminal_voltages)

HERE = Path(__file__).resolve().parent
HEADERS = ('surrogate','probability_source','train_samples','em_components',
           'max_score_error_vs_true_mps','offline_ac_calls','sites','hidden_sites','dense_states','probability_mps_max_rank',
           'probability_mps_scalars','operator_mpo_max_rank',
           'operator_mpo_scalars_per_action','dense_operator_scalars_per_action',
           'operator_fourier_terms','operator_uniform_error_bound',
           'evidence_probability','selected_P','selected_Q','mean_P','mean_Q',
           'max_error_vs_sample_mean','max_monte_carlo_standard_error',
           'max_error_vs_exact_enumeration','linearized_regret_vs_best_sample',
           'linearized_regret_vs_mean_action','conditional_score_gap',
           'selection_certified_by_fourier_bound',
           'ac_terminal_voltage_mae','ac_unsafe_fraction',
           'ac_loss_selected','ac_loss_mean','ac_regret_vs_best_sample',
           'ac_gain_vs_mean','ac_gain_standard_error','ac_gain_ci95_low','ac_gain_ci95_high',
           'operator_build_seconds','model_training_seconds','online_scores_seconds')


def experiment(sites, *, harmonics=16, samples=10000, ac_check=0,
               seed=20261120, surrogate="ac-calibrated",
               probability_source="known-born", train_samples=6000, components=3):
    """Return one row of reproducible diagnostics for a known-law test case."""
    if sites < 3 or harmonics < 1 or samples < 10 or ac_check < 0:
        raise ValueError('sites >= 3, harmonics >= 1, samples >= 10, ac_check >= 0 required')
    local_p, mix = demo_correlations(sites)
    true_psi = correlated_born_mps(local_p, mix)
    training_seconds = 0.
    if probability_source == 'known-born':
        psi = true_psi
        score_function, posterior_function = conditional_scores, posterior
    elif probability_source == 'em-learned':
        t0 = perf_counter()
        training_data = sample_born_mixture(local_p, mix, {}, train_samples,
                            np.random.default_rng(np.random.SeedSequence([seed,sites,1977])))
        fit = fit_mixture_em(training_data,components=components,seed=seed)
        psi = mixture_probability_mps(fit['weights'],fit['local_probabilities'])
        training_seconds = perf_counter() - t0
        score_function, posterior_function = probability_scores, probability_posterior
    else:
        raise ValueError('probability_source must be known-born or em-learned')
    spec = feeder_spec(sites)
    observed = {0:1, sites - 1:1}  # Two remotely measured load sites; others missing.
    t0 = perf_counter()
    if surrogate == 'ac-calibrated':
        models, offline_ac_calls = ac_calibrated_terminal_models(sites,ACTIONS,spec)
    elif surrogate == 'lin-distflow':
        models = tuple(terminal_model(sites, action, spec) for action in ACTIONS)
        offline_ac_calls = 0
    else:
        raise ValueError('surrogate must be ac-calibrated or lin-distflow')
    # One common Fourier interval encloses *every* load/action combination.
    domain = 1.05 * max(max(abs(c + f.min(axis=1).sum()),
                            abs(c + f.max(axis=1).sum())) for c, f in models)
    ops = tuple(fourier_deadband_mpo(c, f, DEADBAND, domain, harmonics)
                for c, f in models)
    build_seconds = perf_counter() - t0
    t0 = perf_counter()
    scores, den = score_function(psi, ops, observed)
    score_true, _ = conditional_scores(true_psi, ops, observed)
    score_fit_error = float(np.max(np.abs(scores-score_true)))
    online_seconds = perf_counter() - t0
    chosen = int(np.argmin(scores))
    means = np.array([float(posterior_function(psi, observed, j) @ np.asarray(LEVELS))
                      for j in range(sites) if j not in observed])
    # For mean plug-in, observed loads use the known categorical level.
    mean_x = np.empty(sites)
    j = 0
    for i in range(sites):
        if i in observed:
            mean_x[i] = LEVELS[observed[i]]
        else:
            mean_x[i] = means[j]
            j += 1
    # The affine model stores f_i at three physical levels. Interpolating at
    # E[P_i|evidence] gives the exact plug-in score for each linear surrogate.
    point_scores = np.array([loss(1 + c + np.sum(f[:, 0] +
                      (f[:, -1]-f[:, 0]) * (mean_x - LEVELS[0]) / (LEVELS[-1]-LEVELS[0])),
                      DEADBAND) for c, f in models])
    mean_action = int(np.argmin(point_scores))
    rng = np.random.default_rng(np.random.SeedSequence([seed, sites]))
    draws = sample_born_mixture(local_p, mix, observed, samples, rng)
    sample_losses = direct_deadband_scores(draws, ACTIONS, spec, DEADBAND, models=models)
    estimated = sample_losses.mean(axis=0)
    stderr = sample_losses.std(axis=0, ddof=1) / np.sqrt(samples)
    exact_error = None
    if sites <= 7:
        # Small-N verification ONLY; never used in operator construction.
        dense_prob = reconstruct(psi)**2 if probability_source=='known-born' else reconstruct(psi)
        grid = np.array(list(product(range(3), repeat=sites)), dtype=int)
        keep = np.array([all(state[i] == v for i, v in observed.items()) for state in grid])
        weights = dense_prob.ravel()[keep]
        weights /= weights.sum()
        exact_losses = direct_deadband_scores(grid[keep], ACTIONS, spec, DEADBAND, models=models)
        exact_scores = weights @ exact_losses
        exact_error = float(np.max(np.abs(scores - exact_scores)))
        if exact_error > fourier_uniform_error_bound(domain, harmonics) + 1e-12:
            raise AssertionError('Fourier uniform bound contradicted by exact enumeration')
    sorted_scores = np.sort(scores)
    gap = float(sorted_scores[1] - sorted_scores[0])
    eps = fourier_uniform_error_bound(domain, harmonics)
    ac_mae, ac_unsafe = None, None
    ac_selected, ac_mean, ac_regret, ac_gain = None, None, None, None
    ac_gain_se, ac_low, ac_high = None, None, None
    if ac_check:
        subset = draws[:ac_check]
        real_losses = []
        for a in ACTIONS:
            volts, violations = full_ac(subset, a, spec)
            real_losses.append(np.maximum(0, np.abs(volts-1.)-DEADBAND))
        real_losses = np.array(real_losses).T
        ac_average = real_losses.mean(axis=0)
        c, f = models[chosen]
        volts, violations = full_ac(subset, ACTIONS[chosen], spec)
        linear_volts = 1 + c + f[np.arange(sites)[None, :], subset].sum(axis=1)
        ac_mae = float(np.mean(np.abs(volts - linear_volts)))
        ac_unsafe = float(np.mean(violations))
        ac_selected = float(ac_average[chosen])
        ac_mean = float(ac_average[mean_action])
        ac_regret = float(ac_average[chosen]-min(ac_average))
        ac_gain = float(ac_average[mean_action]-ac_average[chosen])
        paired_gain = real_losses[:,mean_action] - real_losses[:,chosen]
        ac_gain_se = float(paired_gain.std(ddof=1)/np.sqrt(len(paired_gain)))
        ac_low, ac_high = ac_gain-1.96*ac_gain_se, ac_gain+1.96*ac_gain_se
    result = dict(surrogate=surrogate,probability_source=probability_source,
                  train_samples=train_samples if probability_source=='em-learned' else 0,
                  em_components=components if probability_source=='em-learned' else 0,
                  max_score_error_vs_true_mps=score_fit_error,
                  offline_ac_calls=offline_ac_calls,sites=sites, hidden_sites=sites-len(observed),
                  dense_states=3**sites,
                  probability_mps_max_rank=max(tt_bond_ranks(psi)),
                  probability_mps_scalars=tt_scalar_count(psi),
                  operator_mpo_max_rank=max(tt_bond_ranks(ops[0])),
                  operator_mpo_scalars_per_action=tt_scalar_count(ops[0]),
                  dense_operator_scalars_per_action=3**sites,
                  operator_fourier_terms=harmonics,
                  operator_uniform_error_bound=eps,
                  evidence_probability=den,
                  selected_P=ACTIONS[chosen][0],selected_Q=ACTIONS[chosen][1],
                  mean_P=ACTIONS[mean_action][0],mean_Q=ACTIONS[mean_action][1],
                  max_error_vs_sample_mean=float(np.max(np.abs(scores-estimated))),
                  max_monte_carlo_standard_error=float(np.max(stderr)),
                  max_error_vs_exact_enumeration=exact_error,
                  linearized_regret_vs_best_sample=float(estimated[chosen]-min(estimated)),
                  linearized_regret_vs_mean_action=float(estimated[mean_action]-estimated[chosen]),
                  conditional_score_gap=gap,
                  selection_certified_by_fourier_bound=bool(gap > 2 * eps),
                  ac_terminal_voltage_mae=ac_mae,ac_unsafe_fraction=ac_unsafe,
                  ac_loss_selected=ac_selected,ac_loss_mean=ac_mean,
                  ac_regret_vs_best_sample=ac_regret,ac_gain_vs_mean=ac_gain,
                  ac_gain_standard_error=ac_gain_se,ac_gain_ci95_low=ac_low,
                  ac_gain_ci95_high=ac_high,
                  operator_build_seconds=build_seconds,
                  model_training_seconds=training_seconds,
                  online_scores_seconds=online_seconds)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sites',nargs='+',type=int,default=[4,6,10,16])
    p.add_argument('--harmonics',type=int,default=16)
    p.add_argument('--surrogate',choices=['ac-calibrated','lin-distflow'],
                   default='ac-calibrated')
    p.add_argument('--samples',type=int,default=10000)
    p.add_argument('--probability-source',choices=['known-born','em-learned'],
                   default='known-born')
    p.add_argument('--train-samples',type=int,default=6000)
    p.add_argument('--components',type=int,default=3)
    p.add_argument('--ac-check',type=int,default=200,
                   help='independent full AC validation samples, no AC data used in MPO construction')
    p.add_argument('--seed',type=int,default=20261120)
    p.add_argument('--out',type=Path,default=HERE/'results'/'scaling.csv')
    args=p.parse_args()
    rows=[experiment(s, harmonics=args.harmonics, samples=args.samples,
                     ac_check=args.ac_check,seed=args.seed,
                     surrogate=args.surrogate,probability_source=args.probability_source,
                     train_samples=args.train_samples,components=args.components)
          for s in args.sites]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {args.out}')
    for row in rows:
        print(f"n={row['sites']:2} 3^n={row['dense_states']:>10,} "
              f"MPO={row['operator_mpo_scalars_per_action']:>6,} "
              f"max_rank={row['operator_mpo_max_rank']:>2} "
              f"exact_loss_error={row['max_error_vs_exact_enumeration']} "
              f"AC_validation_MAE={row['ac_terminal_voltage_mae']} "
              f"AC_gain_vs_mean={row['ac_gain_vs_mean']}")


if __name__=='__main__':
    main()
