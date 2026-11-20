"""Chronological ablation of previously received meter reports for voltage dispatch."""
import argparse
import json
from pathlib import Path

import numpy as np

from controller import VoltageController
from coordination_benchmark import Scenario
from measured_profiles import (DATA_PATH, apply_categories, construct_profiles, read_ausgrid,
                               regime, split_indices)
from topologies import LAYOUTS, make_feeder
from topology_benchmark import CONDITIONS, moving_block_interval, paired_effect, validate_actions, voltage_scores


LAGS = {"memoryless": 0, "half_hour": 1, "previous_day": 48}


def available_reports(indices, levels, blackout, alternating=False):
    """Record the telemetry that arrived at each timestamp; -1 denotes no report."""
    indices = np.asarray(indices)
    if (indices.ndim != 1 or len(indices) != len(levels) or not len(indices)
            or np.any(np.diff(indices) != 1)):
        raise ValueError("temporal replay requires aligned consecutive half-hours")
    slot = indices % 336
    missing = (slot >= 116) & (slot < 164) if blackout else np.zeros(len(indices), bool)
    if alternating:
        if blackout or levels.shape[1] != 2:
            raise ValueError("alternating telemetry requires two meters and no total blackout")
        return np.where(indices[:, None] % 2 == np.arange(2), levels, -1), missing
    return np.where(missing[:, None], -1, levels), missing


def lag_reports(reports, lag):
    """At replay start the report buffer is empty, including for the daily lag."""
    previous = np.full_like(reports, -1)
    if 0 < lag < len(reports):
        previous[lag:] = reports[:-lag]
    return previous


def evidence(buses, row):
    return {bus: int(category) for bus, category in zip(buses, row, strict=True) if category >= 0}


def replay_policy(model, layout, indices, reports, lag):
    previous = lag_reports(reports, lag)
    previous_clock = np.where(np.any(previous >= 0, axis=1), regime(indices-lag), -1)
    keys = np.column_stack((regime(indices), previous_clock, reports, previous))
    # The temporal model conditions on current/lagged clock buckets and the report patterns.
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    actions, predicted, inferred = [], [], []
    meters = [layout.sites.index(bus) for bus in layout.meters]
    hidden = [i for i, bus in enumerate(layout.sites) if bus not in layout.meters]
    for i in first:
        bucket = int(regime(indices[i]))
        current = evidence(layout.meters, reports[i])
        past = evidence(layout.meters, previous[i])
        history = {lag: past} if lag and past else {}
        options = {"history": history, "half_hour": int(indices[i])} if history else {}
        policy = "temporal" if history else "adaptive"
        result = model.dispatch(bucket, current, method=policy, **options)
        actions.append(result.actions)
        # One-step predictive score *before* receiving the current reports. Current
        # truth is passed only to the downstream scorer, never to this prediction.
        prior = np.asarray(model.predict_categories(bucket, {}, **options))
        posterior = np.asarray(model.predict_categories(bucket, current, **options)) if current else prior
        predicted.append(prior[meters])
        inferred.append(posterior[hidden])
    actions = np.asarray(actions)
    validate_actions(actions, model, {})
    return actions[inverse], np.asarray(predicted)[inverse], np.asarray(inferred)[inverse], previous, len(first)


def marginal_nll(probabilities, truth):
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=2), 1.)):
        raise ValueError("invalid predictive category probabilities")
    mass = np.take_along_axis(probabilities, truth[:, :, None], axis=2)[:, :, 0]
    if np.any(mass <= 0):
        raise ValueError("observed category had zero probability under fitted model")
    return -np.log(mass).mean(axis=1)


def effects_with_blocks(difference, indices, seed):
    summary = paired_effect(difference, indices, seed)
    summary["moving_block_ci95_by_days"] = {
        length: moving_block_interval(difference, indices, length, seed)
        for length in (7, 34) if len(indices) // 48 > length}
    return summary


def evaluate_temporal(feeder, model, layout, indices, snapshots, *, blackout=False, alternating=False, seed=20261120):
    """Compare three fixed policies on the identical continuous PQ series and device menus."""
    indices = np.asarray(indices)
    if indices[0] % 48 or len(indices) % 48:
        raise ValueError("use complete days for the paired time blocks")
    categories = apply_categories(snapshots, model.thresholds, site_buses=layout.sites)
    meter_truth = categories[:, [layout.sites.index(b) for b in layout.meters]]
    hidden_buses = [bus for bus in layout.sites if bus not in layout.meters]
    hidden_truth = categories[:, [layout.sites.index(b) for b in hidden_buses]]
    reports, missing = available_reports(indices, meter_truth, blackout, alternating)
    metrics, scores, all_actions, history_available = {}, {}, {}, {}
    for method, lag in LAGS.items():
        actions, probabilities, posterior, previous, n = replay_policy(model, layout, indices, reports, lag)
        adjusted = snapshots.copy()
        adjusted[:, np.asarray(layout.controls)-1] -= actions
        voltage, loss, unsafe = voltage_scores(feeder.ac_batch(adjusted), layout.monitors)
        nll = marginal_nll(probabilities, meter_truth)
        hidden_nll = marginal_nll(posterior, hidden_truth)
        history_available[method] = np.any(previous >= 0, axis=1)
        scores[method] = dict(mean_monitor_loss=loss, all_bus_unsafe_fraction=unsafe,
                              pre_report_meter_nll=nll, post_report_hidden_site_nll=hidden_nll)
        all_actions[method] = actions
        metrics[method] = dict(**voltage, pre_report_meter_nll=float(nll.mean()),
                               post_report_hidden_site_nll=float(hidden_nll.mean()),
                               distinct_contexts=n, lag_half_hours=lag,
                               intervals_with_lagged_report=int(history_available[method].sum()))
    paired, deltas = {}, {}
    for method in tuple(LAGS)[1:]:
        deltas[method] = {name: scores["memoryless"][name] - values for name, values in scores[method].items()}
        paired[method] = {name: effects_with_blocks(v, indices, seed) for name, v in deltas[method].items()}
        changed = np.any(all_actions[method] != all_actions["memoryless"], axis=(1, 2))
        if changed[~history_available[method]].any():
            raise AssertionError("empty history must reproduce memoryless dispatch")
        paired[method]["changed_dispatch_intervals"] = int(changed.sum())
    outage = {}
    if missing.any():
        for method in LAGS:
            outage[method] = {key: float(values[missing].mean()) for key, values in scores[method].items()}
            outage[method]["intervals_with_lagged_report"] = int((missing & history_available[method]).sum())
    return dict(intervals=len(indices), no_current_report_intervals=int(missing.sum()),
                reports_received_by_bus={b: int((reports[:, i] >= 0).sum()) for i,b in enumerate(layout.meters)},
                never_metered_buses=hidden_buses,
                methods=metrics, paired_improvement_over_memoryless=paired,
                during_blackout=outage), deltas


def run_temporal(*, test_days=115, seed=20261120, progress=False):
    if not 1 <= test_days <= 115:
        raise ValueError("require 1..115 complete held-out days")
    times, gross, solar = read_ausgrid(DATA_PATH)
    cases, macro = {}, {"primary": {}, "solar_stress": {}, "solar_alternating_meters": {}}
    for topology in ("ieee33", "case69"):
        layout = LAYOUTS[topology]
        lags = tuple(range(0, len(layout.sites)*3, 3))
        for condition in (c for c in CONDITIONS if c.name in ("nominal", "solar_export")):
            feeder = make_feeder(topology, slack=condition.slack, base_load_scale=condition.load)
            powers = construct_profiles(gross, solar, feeder, sites=layout.sites, day_lags=lags,
                                        solar_scale=condition.pv)
            train, test = split_indices(len(powers), day_lags=lags, test_days=test_days)
            scenario = Scenario(condition.name, observed=tuple(layout.sites.index(b) for b in layout.meters))
            model = VoltageController.fit(feeder, scenario, powers[train], train, site_buses=layout.sites,
                                          control_buses=layout.controls, monitor_buses=layout.monitors, seed=seed)
            primary = condition.name == "nominal"
            for telemetry in (("normal_reports", "intermittent_outage") if primary
                              else ("normal_reports", "alternating_reports")):
                alternating = telemetry == "alternating_reports"
                key = topology + "_" + telemetry
                if not primary:
                    key += "_solar_export"
                cases[key], deltas = evaluate_temporal(feeder, model, layout, test, powers[test],
                                                       blackout=telemetry == "intermittent_outage",
                                                       alternating=alternating, seed=seed)
                group = macro["solar_alternating_meters" if alternating else "primary" if primary else "solar_stress"]
                for method, values in deltas.items():
                    for name, delta in values.items():
                        group.setdefault((method, name), []).append(delta)
                if progress:
                    print(key, flush=True)
    aggregate = {name: {m: {} for m in tuple(LAGS)[1:]} for name in macro}
    for group, values in macro.items():
        for (method, name), arrays in values.items():
            aggregate[group][method][name] = effects_with_blocks(np.mean(arrays, axis=0), test, seed)
    return dict(cases=cases, macro_paired_improvement_over_memoryless=aggregate, seed=seed,
                train_first=str(times[train[0]]), train_last=str(times[train[-1]]),
                last_raw_training_time=str(times[train[-1]+48*max(lags)]),
                test_first=str(times[test[0]]), test_last=str(times[test[-1]]),
                protocol=[
                    "Three separate policies share one fitted model per grid and load/PV profile. Four "
                    "primary cases use nominal conditions with/without outages; two supplementary cases "
                    "use the predefined solar_export condition under normal reports. Two diagnostic cases "
                    "alternate which of the two solar-scenario meters reports at each half-hour. Memoryless uses the "
                    "current six-hour bucket and received reports. The two temporal policies additionally use "
                    "received categories from exactly one or 48 half-hours ago, respectively.",
                    "Start each replay with an empty history. During outages no readings enter the buffer. "
                    "No imputation or substitution of older data. All available PCC actions/budgets stay identical.",
                    "For blackout cases, no meter reports arrive when 116 <= timestamp modulo 336 < 164. "
                    "True continuous injections are used only for post-dispatch AC and ground-truth categories.",
                    "Pre-report NLL averages the negative natural log of each of the two meter categories' "
                    "marginal probability before either current meter arrives, in nats per meter. "
                    "Post-report hidden-site NLL scores the ten never-metered sites only, conditional on "
                    "all reports currently and historically available, in nats per site.",
                    "Positive paired difference means lower loss, unsafe frequency, or NLL with history. "
                    "Macro averages cases at each timestamp before resampling whole days or 7/34-day blocks.",
                ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-days", type=int, choices=range(1, 116), default=115, metavar="1..115")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_temporal(test_days=args.test_days, seed=args.seed, progress=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
