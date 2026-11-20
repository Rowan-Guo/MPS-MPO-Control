"""Paired electrical-length sweep. Keep the topology, data, devices and budgets fixed."""
import argparse
import json
from pathlib import Path

import numpy as np

from coordination_benchmark import CONTROL_BUSES, Scenario, TIMESHIFTS, scenario_benchmark
from fast_fourier import FactorizedFourier
from feeder33 import Feeder33, action_injections
from measured_profiles import (DATA_PATH, OBSERVED_SITES, apply_categories, construct_profiles,
                               fit_categories, read_ausgrid, regime, split_indices)
from powerflow import ACTIONS
from probability_tt import fit_mixture_em
from realistic_benchmark import DEADBAND, MONITORED, calibrate_multi_terminal


PROFILE_BUSES = (6, 14, 18, 33)


def weekly_interval(values, times, seed):
    """Paired 7-day block bootstrap, weighted by included half-hour observations."""
    values = np.asarray(values, dtype=float)
    blocks = (times // 48 - times[0] // 48) // 7
    sums = np.bincount(blocks, weights=values)
    counts = np.bincount(blocks)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(sums), size=(2000, len(sums)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return np.quantile(estimates, [.025, .975]).tolist()


def reactive_sensitivities(feeder):
    """Centered AC derivative d|V_k|/dQ_i at the common nominal demand vector."""
    step = .0005
    monitor = np.asarray(PROFILE_BUSES) - 1
    derivatives = {}
    for bus in CONTROL_BUSES:
        up = np.abs(feeder.ac(feeder.nominal_pq, actions=[(bus, 0., step)]))
        down = np.abs(feeder.ac(feeder.nominal_pq, actions=[(bus, 0., -step)]))
        derivatives[str(bus)] = ((up[monitor] - down[monitor]) / (2 * step)).tolist()
    return derivatives


def fixed_menu_oracle(feeder, loads, positive_q_at_33):
    """Best locally feasible original equal-split action, with perfect hindsight."""
    best = np.full(len(loads), np.inf)
    monitor = np.asarray(MONITORED) - 1
    for action in ACTIONS:
        if not positive_q_at_33 and action[1] > 0:
            continue
        v = np.abs(feeder.ac_batch(loads, actions=action_injections(action)))
        losses = np.maximum(np.abs(v[:, monitor] - 1.) - DEADBAND, 0.).mean(axis=1)
        best = np.minimum(best, losses)
    return best


def redirected_dispatch(action):
    """Transfer bus 33's positive Q share to 18. Keep the original total P,Q."""
    p, q = action
    return ((18, p/2, q if q > 0 else q/2), (33, p/2, 0. if q > 0 else q/2))


def redirected_baseline(feeder, powers, times, centers, thresholds, mixtures):
    """Give the six-action controller the same derating information and recalibrate AC."""
    constants, changes, _ = calibrate_multi_terminal(feeder, centers, dispatch_map=redirected_dispatch)
    operator = FactorizedFourier(constants, changes, DEADBAND, 16)
    observed = apply_categories(powers[times], thresholds)[:, OBSERVED_SITES]
    keys = np.column_stack((regime(times), observed))
    commands = np.empty(len(times), dtype=int)
    for key in np.unique(keys, axis=0):
        model = mixtures[key[0]]
        evidence = dict(zip(OBSERVED_SITES, key[1:], strict=True))
        scores = operator.scores(model['weights'], model['local_probabilities'], evidence)
        commands[(keys == key).all(axis=1)] = scores.mean(axis=1).argmin()
    physical = powers[times].copy()
    for i, action in enumerate(ACTIONS):
        for bus, p, q in redirected_dispatch(action):
            physical[commands == i,bus-1] -= [p,q]
    return np.abs(feeder.ac_batch(physical))


def paired_cases(lengths, max_test, seed, gc, pv, powers, positive_q_at_33, baseline_fit):
    results = []
    base_gain = base_times = None
    for length in lengths:
        scenario = Scenario(f"paired_{length:g}x", length=length, reactive_limit_33=not positive_q_at_33)
        metrics, times, voltages = scenario_benchmark(
            scenario, gc, pv, max_test, 16, .0001, seed, return_traces=True)
        fixed = voltages["equal_split_original"]
        distributed = voltages["distributed"]
        fixed_excess = np.maximum(np.abs(fixed - 1.) - DEADBAND, 0.)
        distributed_excess = np.maximum(np.abs(distributed - 1.) - DEADBAND, 0.)
        monitor = np.asarray(MONITORED) - 1
        gain = (fixed_excess - distributed_excess)[:, monitor].mean(axis=1)
        if base_times is None:
            base_gain, base_times = gain, times
        elif not np.array_equal(times, base_times):
            raise RuntimeError("paired sweep requires identical test times")
        amplification = gain - base_gain
        feeder = Feeder33(trunk_length_factor=length)
        oracle = fixed_menu_oracle(feeder, powers[times], positive_q_at_33)
        fixed_loss = fixed_excess[:, monitor].mean(axis=1)
        distributed_loss = distributed_excess[:, monitor].mean(axis=1)
        rerouted = fixed if positive_q_at_33 else redirected_baseline(feeder, powers, times, *baseline_fit)
        rerouted_loss = np.maximum(np.abs(rerouted[:,monitor] - 1.) - DEADBAND, 0.).mean(axis=1)
        rerouted_gain = rerouted_loss - distributed_loss
        fixed_bad = np.any((fixed < .95) | (fixed > 1.05), axis=1).astype(int)
        rerouted_bad = np.any((rerouted < .95) | (rerouted > 1.05), axis=1).astype(int)
        dist_bad = np.any((distributed < .95) | (distributed > 1.05), axis=1).astype(int)
        results.append(dict(
            length_factor=length, intervals=len(times), positive_q_at_33=positive_q_at_33,
            sensitivity_dV_dQ_by_actuator=reactive_sensitivities(feeder),
            fixed_loss=float(fixed_loss.mean()), distributed_loss=float(distributed_loss.mean()),
            original_menu_oracle_loss=float(oracle.mean()),
            original_menu_selection_regret=float((fixed_loss - oracle).mean()),
            rerouted_six_loss=float(rerouted_loss.mean()),
            rerouted_six_any_bus_unsafe=float(rerouted_bad.mean()),
            weekly_ci95_unsafe_fraction_reduction_vs_rerouting=weekly_interval(rerouted_bad-dist_bad,times,seed),
            weekly_ci95_unsafe_fraction_reduction_vs_original=weekly_interval(fixed_bad-dist_bad,times,seed),
            improvement_over_rerouted_six=float(rerouted_gain.mean()),
            weekly_ci95_over_rerouted_six=weekly_interval(rerouted_gain, times, seed),
            improvement_over_original_menu_oracle=float((oracle - distributed_loss).mean()),
            fixed_any_bus_unsafe=metrics["metrics"]["equal_split_original"]["unsafe_fraction"],
            distributed_any_bus_unsafe=metrics["metrics"]["distributed"]["unsafe_fraction"],
            mean_loss_reduction=float(gain.mean()),
            weekly_ci95_reduction=weekly_interval(gain, times, seed),
            additional_gain_vs_1x=float(amplification.mean()),
            weekly_ci95_additional_gain=weekly_interval(amplification, times, seed),
            mean_deadband_excess_reduction_per_profile_bus=
                (fixed_excess - distributed_excess)[:, np.asarray(PROFILE_BUSES)-1].mean(axis=0).tolist(),
            mean_pq_by_actuator=metrics["mean_dispatch_pq"],
        ))
    return results


def run(lengths=(1., 1.5, 2.), max_test=5520, seed=20261120):
    if (not lengths or lengths[0] != 1. or any(not np.isfinite(x) or x <= 0 for x in lengths)
            or any(b <= a for a, b in zip(lengths, lengths[1:])) or max_test < 20):
        raise ValueError("use increasing factors starting at 1 and at least 20 test intervals")
    _, gc, pv = read_ausgrid(DATA_PATH)
    powers = construct_profiles(gc, pv, Feeder33(), day_lags=TIMESHIFTS)
    train, _ = split_indices(len(powers), day_lags=TIMESHIFTS)
    thresholds, centers, states = fit_categories(powers[train])
    mixtures = [fit_mixture_em(states[regime(train) == bucket], components=3,
                               seed=seed+bucket, restarts=2, iterations=85) for bucket in range(4)]
    baseline_fit = centers, thresholds, mixtures
    cases = [row for enabled in (True, False)
             for row in paired_cases(lengths, max_test, seed, gc, pv, powers, enabled, baseline_fit)]
    return dict(profile_buses=PROFILE_BUSES, monitors=MONITORED, seed=seed,
                description="R and X scaled together only on branches 1--18; all laterals retained. "
                            "Fixed original dispatch and independent station dispatch share aggregate budgets. "
                            "Positive additional gain means larger absolute loss reduction than at 1x.",
                cases=cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', nargs='+', type=float, default=(1., 1.5, 2.))
    parser.add_argument('--max-test', type=int, default=5520)
    parser.add_argument('--out', type=Path, default=Path('examples/long_feeder.json'))
    args = parser.parse_args()
    report = run(args.lengths, args.max_test)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf8')
    for case in report['cases']:
        print(case['positive_q_at_33'], case['length_factor'], case['mean_loss_reduction'],
              case['weekly_ci95_additional_gain'])
