"""Measured-profile IEEE33 benchmark for discrete dispatch with local PCC limits.

The policy sees the clock regime and categorical telemetry. Continuous held-out
injections are used only for post-dispatch nonlinear power-flow evaluation.
All policies respect the same fleet P and absolute-Q budget. The centralized
discrete reference additionally has exactly the distributed policy's action set,
quantization, conditional distribution and Fourier objective.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from distributed import Station, solve_dispatch
from fast_fourier import FactorizedFourier
from feeder33 import Feeder33, action_injections
from measured_profiles import (DATA_PATH, OBSERVED_SITES, SITES, apply_categories,
    construct_profiles, fit_categories, read_ausgrid, regime, split_indices, state_power)
from powerflow import ACTIONS
from probability_tt import fit_mixture_em
from realistic_benchmark import DEADBAND, MONITORED, block_bootstrap_ci, calibrate_multi_terminal


CONTROL_BUSES = (14, 18, 33)
STEP = 0.025
BUDGET = (2, 2)  # total active steps and total absolute reactive steps
GUARD_BAND = 0.045  # 0.005 p.u. buffer inside the all-bus operational limits
LOCAL_GRID = tuple(product(range(3), range(-2, 3)))
TIMESHIFTS = tuple(range(0, len(SITES) * 3, 3))


@dataclass(frozen=True)
class Scenario:
    name: str
    length: float = 1.0
    load: float = 0.65
    pv: float = 1.5
    slack: float = 1.02
    reactive_limit_33: bool = False
    observed: tuple[int, ...] = OBSERVED_SITES


SCENARIOS = (
    Scenario("original"),
    Scenario("trunk_1.5x", length=1.5),
    Scenario("trunk_2x", length=2.0),
    Scenario("heavy_demand", load=0.95, pv=0.8),
    Scenario("high_pv", load=0.65, pv=3.0, slack=1.035),
    Scenario("bus33_no_positive_q", reactive_limit_33=True),
    Scenario("one_meter", observed=(2,)),
    Scenario("four_meters", observed=(2, 4, 9, 10)),
)


def station_tables(feeder, reference_loads, scenario):
    """Exact one-station AC response at the common reference operating point."""
    base = np.abs(feeder.ac(reference_loads))
    stations = []
    for bus in CONTROL_BUSES:
        grid = [(p, q) for p, q in LOCAL_GRID
                if not (bus == 33 and scenario.reactive_limit_33 and q > 0)]
        actions = [(p * STEP, q * STEP) for p, q in grid]
        injections = np.repeat(reference_loads[None], len(actions), axis=0)
        injections[:, bus - 1] -= actions
        shifts = np.abs(feeder.ac_batch(injections)) - base
        usage = [(p, abs(q)) for p, q in grid]
        stations.append(Station(bus=bus, actions=actions, voltage_shifts=shifts,
                                costs=np.zeros(len(grid)), resources=usage))
    return stations


def reference_actions(stations, tick):
    """Independent exhaustive reference; only used to verify the distributed solve."""
    options = np.asarray(list(product(*(range(len(s.actions)) for s in stations))))
    resources = sum(np.asarray(s.resources)[options[:, i]] for i, s in enumerate(stations))
    eligible = np.all(resources <= BUDGET, axis=1)
    for i, s in enumerate(stations):
        if s.available is not None:
            eligible &= np.asarray(s.available)[options[:, i]]
    options = options[eligible]
    shift = sum(np.rint(np.asarray(s.voltage_shifts)[options[:, i]] / tick).astype(int)
                for i, s in enumerate(stations))
    return options, shift * tick


def feasible_equal_splits(stations, tick):
    """Map the six legacy presets to local action indices; enforce live PCC limits."""
    choices, original_indices, shifts = [], [], []
    for j, (p, q) in enumerate(ACTIONS):
        target = [(p / 2, q / 2) if s.bus in (18, 33) else (0., 0.) for s in stations]
        try:
            choice = tuple(list(map(tuple, s.actions)).index(pair)
                           for s, pair in zip(stations, target, strict=True))
        except ValueError:  # unavailable positive Q, for example
            continue
        if any(s.available is not None and not s.available[a] for s,a in zip(stations,choice,strict=True)):
            continue
        if np.any(sum(np.asarray(s.resources)[a] for s,a in zip(stations,choice,strict=True)) > BUDGET):
            continue
        shift = sum(np.rint(np.asarray(s.voltage_shifts)[a] / tick).astype(int)
                    for s,a in zip(stations,choice,strict=True))
        shifts.append(shift * tick)
        choices.append(choice)
        original_indices.append(j)
    if not choices:
        raise ValueError("no feasible equal-split reference, including zero dispatch")
    return np.asarray(original_indices), np.asarray(choices), np.asarray(shifts)


def conditional_voltage_mean(fit, evidence, offsets, contributions):
    """Exact affine conditional mean under the same product mixture as the MPO."""
    local = fit["local_probabilities"]
    weights = fit["weights"].copy()
    for site, category in evidence.items():
        weights *= local[:, site, category]
    weights /= weights.sum()
    marginals = np.einsum("c,cid->id", weights, local)
    for site, category in evidence.items():
        marginals[site] = 0.
        marginals[site, category] = 1.
    return offsets + np.einsum("mid,id->m", contributions, marginals)


def evaluate(feeder, snapshot, stations, choices):
    powers = snapshot.copy()
    for i, station in enumerate(stations):
        powers[:, station.bus - 1] -= np.asarray(station.actions)[choices[:, i]]
    voltages = np.abs(feeder.ac_batch(powers))
    losses = np.maximum(np.abs(voltages[:, np.asarray(MONITORED) - 1] - 1.) - DEADBAND, 0.).mean(axis=1)
    unsafe = np.any((voltages < .95) | (voltages > 1.05), axis=1)
    metrics = dict(mean_loss=float(losses.mean()), p95_loss=float(np.quantile(losses, .95)),
                   worst_loss=float(losses.max()), unsafe_fraction=float(unsafe.mean()),
                   minimum_voltage=float(voltages.min()), maximum_voltage=float(voltages.max()))
    return losses, metrics, voltages


def empirical_deadband(values, band=DEADBAND):
    """O(log N) empirical expected hinge per monitor for arbitrary voltage shifts."""
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("at least one finite training voltage vector is required")
    ordered = np.sort(values, axis=0).T
    sums = np.pad(np.cumsum(ordered, axis=1), ((0, 0), (1, 0)))
    n = len(values)
    def evaluate_shifts(shifts):
        result = np.zeros(len(shifts))
        for j, sample in enumerate(ordered):
            upper = band - shifts[:, j]
            lower = -band - shifts[:, j]
            left = np.searchsorted(sample, lower)
            right = np.searchsorted(sample, upper)
            result += (left * lower - sums[j, left] + sums[j, -1] - sums[j, right]
                       - (n - right) * upper) / n
        return result / len(ordered)
    return evaluate_shifts


def reference_envelope(reference_shift, reference_voltages):
    """Allow no new predicted violations on previously safe training snapshots.

    Anchor the affine *difference* in voltage shifts to exact AC voltages
    under the feasible reference dispatch. Leave a 0.005 p.u. guard buffer
    where the reference already has that headroom. The reference is feasible
    by construction; realized held-out safety is still evaluated separately.
    """
    safe = np.all((reference_voltages >= .95) & (reference_voltages <= 1.05), axis=1)
    if not np.any(safe):
        return lambda shifts: np.zeros(len(shifts))
    rows = reference_voltages[safe]
    low = reference_shift + np.minimum(0., (1 - GUARD_BAND) - rows.min(axis=0))
    high = reference_shift + np.maximum(0., (1 + GUARD_BAND) - rows.max(axis=0))
    def penalty(shifts):
        feasible = np.all((shifts >= low - 1e-12) & (shifts <= high + 1e-12), axis=1)
        return np.where(feasible, 0., np.inf)
    return penalty


def paired_safety_guard(reference_shift, reference_voltages):
    """Screen joint all-bus event increases on matched training records.

    Use exact AC for the reference, an incremental affine prediction for a
    trial. Check the operational band plus a 0.002-p.u. inner buffer. An
    action that introduces events must rescue enough other records to cover
    a one-sided paired count allowance (1.645 sqrt(number of discordances)).
    This is a conservative empirical screen, not a coverage guarantee.
    """
    centered = reference_voltages - 1.
    limits = (.048, .05)
    reference_bad = [np.any(np.abs(centered) > b + 1e-12, axis=1) for b in limits]
    minimum = centered.min(axis=0)
    maximum = centered.max(axis=0)
    def penalty(shifts):
        allowed = np.ones(len(shifts), dtype=bool)
        chunk = max(1, min(64, 200_000 // len(centered)))
        for start in range(0, len(shifts), chunk):
            delta = shifts[start:start + chunk] - reference_shift
            keep = allowed[start:start + len(delta)]
            for band, original in zip(limits, reference_bad, strict=True):
                critical = (minimum + delta.min(axis=0) < -band) | (maximum + delta.max(axis=0) > band)
                now = np.zeros((len(delta), len(centered)), dtype=bool)
                for bus in np.flatnonzero(critical):
                    values = centered[None, :, bus] + delta[:, None, bus]
                    now |= np.abs(values) > band + 1e-12
                new = np.sum(now & ~original, axis=1)
                rescued = np.sum(~now & original, axis=1)
                keep &= (new == 0) | (new - rescued + 1.645*np.sqrt(new + rescued) <= 0)
        return np.where(allowed, 0., np.inf)
    return penalty


def risk_functions(op, fit, evidence, training_voltages, matching, bucket_rows):
    """Condition both estimators on identical telemetry, with a fixed TT prior."""
    evaluator = op.shift_evaluator(fit["weights"], fit["local_probabilities"], evidence)
    monitors = np.asarray(MONITORED) - 1
    def prior(shifts):
        return evaluator(np.asarray(shifts)[:, None, monitors])[:, 0, :].mean(axis=1)

    n = np.count_nonzero(matching)
    rows = matching if n else bucket_rows
    training = training_voltages[rows]
    historical_curve = empirical_deadband(training[:,monitors])
    protection_curve = empirical_deadband(training, GUARD_BAND)
    def historical(shifts):
        return historical_curve(shifts[:,monitors])
    weight = 32 / (32 + n)
    def combined(shifts):
        return weight * prior(shifts) + (1 - weight) * historical(shifts)
    def guard(shifts):
        # Sum the expected per-bus excess beyond the inner guard band.
        return training.shape[1] * protection_curve(shifts)
    def risk(shifts):
        return combined(shifts) + guard(shifts)
    return risk, combined, prior, historical, guard


@dataclass
class ContextRisk:
    risk: Callable
    strict_risk: Callable
    soft_risk: Callable
    no_guard: Callable
    model_risk: Callable
    empirical: Callable
    guard: Callable
    envelope: Callable
    reference_indices: tuple
    envelope_cases: int


def reference_magnitude_table(feeder, training_powers):
    """Offline AC calibration on training intervals only (six shared baselines)."""
    return {j: np.abs(feeder.ac_batch(training_powers, actions=action_injections(a)))
            for j, a in enumerate(ACTIONS)}


def context_risk(stations, fixed_op, op, fit, evidence, states,
                 training_reference_magnitudes, training_buckets, bucket, tick):
    """Prepare terminal costs for one evidence state without visiting joint actions."""
    original_ids, choices, shifts = feasible_equal_splits(stations, tick)
    scores = fixed_op.scores(fit["weights"], fit["local_probabilities"], evidence).mean(axis=1)
    ref = np.argmin(scores[original_ids])
    prior_rows = training_buckets == bucket
    if not np.any(prior_rows):
        raise ValueError("training bucket is empty")
    matching = prior_rows.copy()
    for site, category in evidence.items():
        matching &= states[:, site] == category
    training_voltages = training_reference_magnitudes[ACTIONS.index((0.,0.))] - 1
    soft, combined, model, empirical, guard = risk_functions(op, fit, evidence, training_voltages,
                                                            matching, prior_rows)
    rows = matching if np.any(matching) else prior_rows
    reference = training_reference_magnitudes[original_ids[ref]][rows]
    envelope = reference_envelope(shifts[ref], reference)
    paired = paired_safety_guard(shifts[ref], reference)
    def protected(trial):
        return combined(trial) + paired(trial)
    def strict(trial):
        return soft(trial) + envelope(trial)
    return ContextRisk(protected, strict, soft, combined, model, empirical, guard, paired,
                       tuple(map(int, choices[ref])), len(reference))


def build_shift_operator(c, f, stations, tick, harmonics):
    """Bound the Fourier interval from local shifts without joint enumeration."""
    hi = sum(np.rint(np.asarray(s.voltage_shifts) / tick).max(axis=0) * tick for s in stations)
    lo = sum(np.rint(np.asarray(s.voltage_shifts) / tick).min(axis=0) * tick for s in stations)
    monitors = np.asarray(MONITORED) - 1
    bound = max(abs(c + lo[monitors] + f.min(axis=2).sum(axis=1)).max(),
                abs(c + hi[monitors] + f.max(axis=2).sum(axis=1)).max())
    return FactorizedFourier(c[None], f[None], DEADBAND, harmonics,
                             domain=max(1.05 * bound, 1.01 * DEADBAND))


def scenario_benchmark(scenario, gc, pv, max_test, harmonics, tick, seed,
                       verify_process=False, *, return_traces=False):
    clock = perf_counter()
    feeder = Feeder33(slack=scenario.slack, base_load_scale=scenario.load,
                      trunk_length_factor=scenario.length)
    powers = construct_profiles(gc, pv, feeder, day_lags=TIMESHIFTS, solar_scale=scenario.pv)
    train, test = split_indices(len(powers), day_lags=TIMESHIFTS)
    thresholds, centers, states = fit_categories(powers[train])
    chosen = np.linspace(0, len(test) - 1, min(max_test, len(test)), dtype=int)
    heldout = test[chosen]
    observed_states = apply_categories(powers[heldout], thresholds)
    fits = [fit_mixture_em(states[regime(train) == bucket], components=3,
                           seed=seed + bucket, restarts=2, iterations=85)
            for bucket in range(4)]
    training_seconds = perf_counter() - clock

    clock = perf_counter()
    full_c, full_f, offline_solves = calibrate_multi_terminal(feeder, centers)
    fixed_op = FactorizedFourier(full_c, full_f, DEADBAND, harmonics)
    calibrated_seconds = perf_counter() - clock
    clock = perf_counter()
    no_action = ACTIONS.index((0., 0.))
    c, f = full_c[no_action], full_f[no_action]
    reference = state_power(feeder.nominal_pq, centers, SITES, np.ones(len(SITES), dtype=int))
    stations = station_tables(feeder, reference, scenario)
    options, shifts = reference_actions(stations, tick)
    action_lookup = {tuple(row):i for i,row in enumerate(options)}
    _, equal_choices, _ = feasible_equal_splits(stations,tick)
    equal_ids = np.asarray([action_lookup[tuple(row)] for row in equal_choices])
    monitors = np.asarray(MONITORED) - 1
    op = build_shift_operator(c, f, stations, tick, harmonics)
    setup_seconds = perf_counter() - clock

    # Training observations retain the variability inside each categorical bin.
    # The conditional empirical distribution and the TT prior use the same
    # clock/telemetry key. All six preset training AC tables are precomputed.
    clock = perf_counter()
    reference_magnitudes = reference_magnitude_table(feeder, powers[train])
    calibration_seconds = perf_counter() - clock
    training_buckets = regime(train)
    methods = ("equal_split_original", "equal_split_shared_model", "joint_mean",
               "joint_tt_uncalibrated", "joint_empirical", "joint_without_guard",
               "joint_strict_guard", "distributed")
    decisions = {method: np.empty((len(heldout), len(stations)), dtype=int) for method in methods}
    policy_cache = {}
    max_gap = max_messages = max_wire = 0
    distributed_seconds = centralized_seconds = evidence_seconds = 0.
    process_verified = False
    envelope_training_cases = 0
    for i, (bucket, cat) in enumerate(zip(regime(heldout), observed_states, strict=True)):
        key = (int(bucket), tuple(int(cat[idx]) for idx in scenario.observed))
        if key not in policy_cache:
            clock = perf_counter()
            evidence = dict(zip(scenario.observed, key[1], strict=True))
            fit = fits[bucket]
            ctx = context_risk(stations, fixed_op, op, fit, evidence, states, reference_magnitudes,
                               training_buckets, bucket, tick)
            envelope_training_cases += ctx.envelope_cases
            evidence_seconds += perf_counter() - clock
            clock = perf_counter()
            result = solve_dispatch(stations, ctx.risk, tick=tick, resource_limits=BUDGET)
            distributed_seconds += perf_counter() - clock
            if verify_process and not process_verified:
                replay = solve_dispatch(stations, ctx.risk, tick=tick, resource_limits=BUDGET, transport="process")
                if replay.action_indices != result.action_indices or abs(replay.objective-result.objective) > 1e-12:
                    raise AssertionError("cross-process dispatch disagrees with local message transport")
                process_verified = True
            clock = perf_counter()
            costs = ctx.risk(shifts)
            gap = abs(result.objective - costs.min())
            if gap > 1e-10:
                raise AssertionError(f"distributed objective disagrees with exhaustive search: {gap:g}")
            max_gap = max(max_gap, float(gap))
            centralized_seconds += perf_counter() - clock
            max_messages = max(max_messages, result.diagnostics["messages"])
            max_wire = max(max_wire, result.diagnostics["wire_bytes"])
            # Score original six using their own action-specific AC calibration,
            # then also on our single shared voltage model to isolate flexibility.
            shared = equal_ids[np.argmin(costs[equal_ids])]
            penalty = ctx.envelope(shifts)
            raw = int(np.argmin(ctx.model_risk(shifts) + penalty))
            historical = int(np.argmin(ctx.empirical(shifts) + penalty))
            unguarded = int(np.argmin(ctx.no_guard(shifts)))
            strict = int(np.argmin(ctx.strict_risk(shifts)))
            mean = conditional_voltage_mean(fit, evidence, c, f)
            mean_voltage = shifts[:,monitors] + mean
            plug = np.maximum(np.abs(mean_voltage) - DEADBAND, 0.).mean(axis=1) + penalty
            # Among exact zero-risk ties, smaller mean voltage deviation is a
            # deployable deterministic secondary objective for the plug-in.
            plug_id = np.lexsort((np.abs(mean_voltage).sum(axis=1), plug))[0]
            policy_cache[key] = (ctx.reference_indices, options[shared], options[plug_id],
                                 options[raw], options[historical], options[unguarded], options[strict],
                                 result.action_indices)
        for method, action_ids in zip(methods, policy_cache[key], strict=True):
            decisions[method][i] = action_ids

    realized, metrics, voltage_traces = {}, {}, {}
    for method in methods:
        realized[method], metrics[method], voltages = evaluate(feeder, powers[heldout], stations, decisions[method])
        if return_traces:
            voltage_traces[method] = voltages
    differences = {}
    for method in methods[:-1]:
        delta = realized[method] - realized["distributed"]
        ci = block_bootstrap_ci(delta, heldout // 48, seed=seed)
        baseline_loss = float(realized[method].mean())
        differences[method] = dict(mean_reduction=float(delta.mean()),
            reduction_percent=float(100 * delta.mean() / baseline_loss) if baseline_loss > 1e-12 else None,
            daily_bootstrap_ci95=list(ci))
    choices = decisions["distributed"]
    row = dict(scenario=asdict(scenario), samples=len(heldout), training_intervals=len(train),
        observed_buses=[SITES[i] for i in scenario.observed], cross_process_replay=process_verified,
        feasible_joint_actions=len(options), feasible_original_actions=len(equal_ids),
        contexts=len(policy_cache), control_buses=CONTROL_BUSES, terminal_grid_pu=tick,
        max_central_objective_gap=max_gap, max_messages_per_dispatch=max_messages,
        max_serialized_message_bytes_per_dispatch=max_wire, metrics=metrics,
        paired_improvement_vs=differences,
        mean_dispatch_pq={str(s.bus):np.asarray(s.actions)[choices[:, i]].mean(axis=0).tolist()
                          for i,s in enumerate(stations)},
        original_offline_ac_cases=offline_solves,
        # The original preset bank also supplies the operable safety baseline.
        # The no-action state-response model is reused directly from that bank.
        distributed_affine_calibration_ac_cases=offline_solves+1+sum(len(s.actions) for s in stations),
        empirical_and_reference_offline_ac_cases=len(train)*len(ACTIONS),
        online_ac_calls_per_dispatch=0, envelope_training_rows_consulted=envelope_training_cases,
        timings_seconds=dict(profile_and_probability_fit=training_seconds,
            original_calibration_and_fourier=calibrated_seconds,
            empirical_training_ac=calibration_seconds,
            station_setup_including_reference_enumeration=setup_seconds,
            conditional_risk_build_total=evidence_seconds, all_unique_distributed_solves=distributed_seconds,
            vectorized_bruteforce_scoring_total=centralized_seconds))
    return (row, heldout, voltage_traces) if return_traces else row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-test", type=int, default=320)
    parser.add_argument("--harmonics", type=int, default=16)
    parser.add_argument("--tick", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--scenarios", nargs="+", choices=[s.name for s in SCENARIOS])
    parser.add_argument("--verify-process", action="store_true",
                        help="repeat the first dispatch in every scenario using separate station processes")
    parser.add_argument("--out", type=Path)
    args=parser.parse_args()
    if args.max_test < 20 or args.harmonics < 1 or args.tick <= 0:
        parser.error("max-test >= 20, harmonics >= 1 and positive tick required")
    _, gc, pv = read_ausgrid(DATA_PATH)
    results=[]
    for scenario in SCENARIOS:
        if args.scenarios and scenario.name not in args.scenarios:
            continue
        row=scenario_benchmark(scenario,gc,pv,args.max_test,args.harmonics,args.tick,args.seed,args.verify_process)
        results.append(row)
        print(scenario.name,{k:round(v["mean_loss"],7) for k,v in row["metrics"].items()},flush=True)
    report=dict(seed=args.seed,harmonics=args.harmonics,max_test=args.max_test,tick_pu=args.tick,
                all_bus_paired_guard_bands_pu=[.048,.05], paired_discordance_allowance=1.645,
                strict_ablation_guard_band_pu=GUARD_BAND, empirical_prior_intervals=32,
                fleet_limits_pu=dict(sum_p=STEP*BUDGET[0],sum_abs_q=STEP*BUDGET[1]),
                scenarios=results)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
