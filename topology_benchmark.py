"""Paired, continuous-AC evaluation across radial topologies and operating conditions."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from controller import VoltageController
from coordination_benchmark import BUDGET, STEP, Scenario, reference_actions
from measured_profiles import (DATA_PATH, apply_categories, construct_profiles,
                               read_ausgrid, regime, split_indices)
from realistic_benchmark import DEADBAND, block_bootstrap_ci
from topologies import IEEE33_LAYOUT, LAYOUTS, make_feeder


@dataclass(frozen=True)
class Condition:
    name: str
    load: float = .65
    pv: float = 1.5
    slack: float = 1.02
    derated: bool = False
    meter_outage: bool = False
    time_varying: bool = False


CONDITIONS = (
    Condition("nominal"),
    Condition("pcc_derating", derated=True),
    Condition("telemetry_outage", meter_outage=True),
    Condition("outage_derating_recovery", time_varying=True),
    Condition("heavy_demand", load=.90, pv=.8),
    Condition("solar_export", load=.65, pv=7.5, slack=1.045),
)
METHODS = ("equal_split", "legacy", "adaptive")


def head_active_power(feeder, voltages):
    """Slack real-power export into the feeder (negative when the feeder exports)."""
    children = np.flatnonzero(feeder.parent == 0)
    current = ((voltages[:, :1] - voltages[:, children]) / feeder.impedance[children]).sum(axis=1)
    return (voltages[:, 0] * current.conj()).real


def validate_actions(actions, model, limits):
    for station, choices in zip(model.stations, actions.transpose(1, 0, 2), strict=True):
        available = {tuple(pq) for i, pq in enumerate(station.actions)
                     if station.available is None or station.available[i]}
        if not set(map(tuple, choices)).issubset(available):
            raise AssertionError(f"used an unavailable setpoint at bus {station.bus}")
        if station.bus in limits:
            low, high = limits[station.bus]
            if np.any((choices[:, 1] < low) | (choices[:, 1] > high)):
                raise AssertionError(f"reactive dispatch exceeds the live limits at bus {station.bus}")
    usage = np.stack((actions[:, :, 0].sum(axis=1), abs(actions[:, :, 1]).sum(axis=1)), axis=1)
    if np.any(usage > STEP * np.asarray(BUDGET) + 1e-12):
        raise AssertionError("fleet dispatch exceeds the P or absolute-Q budget")


def operating_masks(condition, indices):
    """Temporary events use only the current simulation clock; 336 half-hours per week."""
    if condition.time_varying:
        slot = np.asarray(indices) % (7 * 48)
        return (slot >= 116) & (slot < 164), (slot >= 132) & (slot < 204)
    return np.full(len(indices), condition.meter_outage), np.full(len(indices), condition.derated)


def dispatch_intervals(model, layout, heldout, powers, condition):
    categories = apply_categories(powers, model.thresholds, site_buses=layout.sites)
    outage, derated = operating_masks(condition, heldout)
    columns = [layout.sites.index(bus) for bus in layout.meters]
    # Hidden categories never enter a context key, including when the link later recovers.
    reports = np.where(outage[:, None], -1, categories[:, columns])
    keys = np.column_stack((outage, derated, regime(heldout), reports))
    contexts, inverse = np.unique(keys, axis=0, return_inverse=True)
    q_cap = {layout.controls[-1]: (-2 * STEP, 0.)}
    tables = {method: [] for method in METHODS}
    diagnostics = {m: {"seconds": 0., "messages": 0, "wire_bytes": 0, "raw_wire_bytes": 0}
                   for m in METHODS}
    selected_policies = []
    for key in contexts:
        bucket = int(key[2])
        limits = q_cap if key[1] else {}
        observed = {} if key[0] else dict(zip(layout.meters, map(int, key[3:]), strict=True))
        for method in METHODS:
            t0 = perf_counter()
            if method == "equal_split":
                actions = model.reference_dispatch(bucket, observed, q_limits=limits)
            else:
                result = model.dispatch(bucket, observed, q_limits=limits, method=method)
                actions = result.actions
                for field in ("messages", "wire_bytes"):
                    diagnostics[method][field] += result.diagnostics[field]
                diagnostics[method]["raw_wire_bytes"] += result.diagnostics.get(
                    "raw_wire_bytes", result.diagnostics["wire_bytes"])
                if method == "adaptive":
                    selected_policies.append(result.diagnostics.get("selected_policy", "incumbent"))
            diagnostics[method]["seconds"] += perf_counter() - t0
            tables[method].append(actions)
    actions = {}
    for method, table in tables.items():
        table = np.asarray(table)
        for flag in np.unique(contexts[:, 1]):
            validate_actions(table[contexts[:, 1] == flag], model, q_cap if flag else {})
        actions[method] = table[inverse]
        counter = diagnostics[method]
        counter["ms_per_unique_context"] = 1000 * counter.pop("seconds") / len(contexts)
        counter["neighbor_messages_per_unique_context"] = counter.pop("messages") / len(contexts)
        counter["wire_bytes_per_unique_context"] = counter.pop("wire_bytes") / len(contexts)
        counter["raw_wire_bytes_per_unique_context"] = counter.pop("raw_wire_bytes") / len(contexts)
    modes = []
    for missing, restricted in np.unique(keys[:, :2], axis=0):
        modes.append(dict(
            intervals=int(((outage == missing) & (derated == restricted)).sum()),
            unique_contexts=int(np.all(contexts[:, :2] == [missing, restricted], axis=1).sum()),
            observed_buses=() if missing else layout.meters, q_limits_pu=q_cap if restricted else {}))
    evidence = dict(unique_contexts=len(contexts), operating_modes=modes)
    labels = np.asarray(selected_policies)
    evidence["adaptive_selected_policy"] = {
        str(label): dict(unique_contexts=int((labels == label).sum()), intervals=int((labels[inverse] == label).sum()))
        for label in np.unique(labels)}
    if len(modes) == 1:
        evidence.update(observed_buses=modes[0]["observed_buses"], q_limits_pu=modes[0]["q_limits_pu"])
    return actions, diagnostics, evidence


def voltage_scores(phasors, monitors):
    mag = abs(phasors)
    monitor_loss = np.maximum(abs(mag[:, np.asarray(monitors) - 1] - 1.) - DEADBAND, 0.).mean(axis=1)
    low = np.maximum(.95 - mag.min(axis=1), 0.)
    high = np.maximum(mag.max(axis=1) - 1.05, 0.)
    excursion = np.maximum(low, high)
    metrics = dict(mean_monitor_loss=float(monitor_loss.mean()),
                   p95_monitor_loss=float(np.quantile(monitor_loss, .95)),
                   all_bus_unsafe_fraction=float((excursion > 0).mean()),
                   all_bus_undervoltage_fraction=float((low > 0).mean()),
                   all_bus_overvoltage_fraction=float((high > 0).mean()),
                   worst_all_bus_limit_excess=float(excursion.max()),
                   min_voltage=float(mag.min()), max_voltage=float(mag.max()))
    return metrics, monitor_loss, (excursion > 0).astype(float)


def paired_effect(difference, indices, seed):
    """Positive values are an improvement of adaptive over its comparator."""
    return dict(mean=float(np.mean(difference)),
                day_block_ci95=block_bootstrap_ci(difference, indices // 48, seed=seed))


def moving_block_interval(difference, indices, block_days, seed):
    """Resample contiguous circular blocks of complete days on the common test clock."""
    days, counts = np.unique(indices // 48, return_counts=True)
    if np.any(counts != 48) or np.any(np.diff(indices) != 1) or not 1 <= block_days <= len(days):
        raise ValueError("moving blocks require consecutive complete days and a supported block length")
    means = np.asarray(difference).reshape(len(days), 48).mean(axis=1)
    rng = np.random.default_rng(seed)
    starts = rng.integers(len(days), size=(500, (len(days) + block_days - 1) // block_days, 1))
    sampled_days = ((starts + np.arange(block_days)) % len(days)).reshape(500, -1)[:, :len(days)]
    return np.quantile(means[sampled_days].mean(axis=1), [.025, .975]).tolist()


def capacity_oracle(feeder, model, snapshots, heldout, decisions):
    """Full-information, offline AC feasibility bound on five prespecified complete days."""
    days = np.unique(heldout // 48)
    sampled_days = days[np.linspace(0, len(days)-1, min(5, len(days)), dtype=int)]
    selected = np.isin(heldout // 48, sampled_days)
    test = snapshots[selected]
    buses = np.array([s.bus for s in model.stations]) - 1

    def safe(actions):
        loads = test.copy()
        loads[:, buses] -= actions
        v = np.abs(feeder.ac_batch(loads))
        return ((v >= .95) & (v <= 1.05)).all(axis=1)

    combinations, _ = reference_actions(model.stations, model.tick)
    recoverable = np.zeros(len(test), bool)
    for choice in combinations:
        action = np.array([s.actions[idx] for s, idx in zip(model.stations, choice, strict=True)])
        recoverable |= safe(action)
    counts = {}
    for method, dispatch in decisions.items():
        is_safe = safe(dispatch[selected])
        if np.any(is_safe & ~recoverable):
            raise AssertionError("a realized feasible dispatch falls outside the oracle action set")
        counts[method] = dict(safe=int(is_safe.sum()), unsafe_but_feasible=int((~is_safe & recoverable).sum()),
                              unsafe_for_every_feasible_action=int((~recoverable).sum()))
    return dict(test_day_offsets=(sampled_days - days[0]).tolist(), intervals=len(test),
                feasible_joint_commands=len(combinations), counts=counts,
                protocol="Small-fleet offline diagnostic. Test every budget-feasible station combination "
                         "with nonlinear AC using the true continuous injections on five evenly spaced test days. "
                         "Safe means every bus is in [0.95,1.05]. Counts refer only to these intervals.")


def evaluate_case(feeder, model, layout, heldout, snapshots, condition, seed):
    actions, timings, evidence = dispatch_intervals(model, layout, heldout, snapshots, condition)
    original = feeder.ac_batch(snapshots)
    head = head_active_power(feeder, original)
    uncontrolled = voltage_scores(original, layout.monitors)[0]
    policy_metrics, losses, unsafe = {}, {}, {}
    for method, decisions in actions.items():
        loads = snapshots.copy()
        loads[:, np.asarray(layout.controls) - 1] -= decisions
        phasors = feeder.ac_batch(loads)
        policy_metrics[method], losses[method], unsafe[method] = voltage_scores(phasors, layout.monitors)
        policy_metrics[method].update(timings[method])
    paired, differences = {}, {}
    for baseline in METHODS[:-1]:
        differences[baseline] = {
            "mean_monitor_loss": losses[baseline] - losses["adaptive"],
            "all_bus_unsafe_fraction": unsafe[baseline] - unsafe["adaptive"],
        }
        paired[baseline] = {name: paired_effect(v, heldout, seed)
                            for name, v in differences[baseline].items()}
        paired[baseline]["different_dispatch_fraction"] = float(
            np.any(actions[baseline] != actions["adaptive"], axis=(1, 2)).mean())
    case = dict(condition=condition.name, load_scale=condition.load, pv_scale=condition.pv,
                slack=condition.slack, **evidence,
                uncontrolled=dict(**uncontrolled, head_export_intervals=int((head < 0).sum()),
                                  head_export_fraction=float((head < 0).mean()),
                                  minimum_head_active_power_pu=float(head.min())),
                policies=policy_metrics, paired_improvement_over=paired)
    if layout == LAYOUTS["case69"] and condition.name == "heavy_demand":
        case["capacity_oracle_subset"] = capacity_oracle(feeder, model, snapshots, heldout, actions)
    return case, differences, actions["adaptive"]


def summarize_policies(grids):
    cases = [case for grid in grids.values() for case in grid["cases"]]
    contexts = [case["unique_contexts"] for case in cases]
    summary = {}
    for method in METHODS:
        metrics = [case["policies"][method] for case in cases]
        summary[method] = {field: float(np.mean([m[field] for m in metrics]))
                           for field in ("mean_monitor_loss", "all_bus_unsafe_fraction")}
        summary[method]["average_case_p95_monitor_loss"] = float(np.mean([m["p95_monitor_loss"] for m in metrics]))
        for field in ("ms_per_unique_context", "neighbor_messages_per_unique_context",
                      "wire_bytes_per_unique_context", "raw_wire_bytes_per_unique_context"):
            summary[method][field] = float(np.average([m[field] for m in metrics], weights=contexts))
    return summary


def run_suite(*, topology_ids=tuple(LAYOUTS), test_days=115, harmonics=16, seed=20261120,
              profile_path=DATA_PATH, progress=False):
    if not 1 <= test_days <= 115 or harmonics < 1:
        raise ValueError("require 1..115 held-out days and positive harmonics")
    topology_ids = tuple(topology_ids)
    if not topology_ids or len(set(topology_ids)) != len(topology_ids):
        raise ValueError("require unique topologies")
    times, gross, solar = read_ausgrid(profile_path)
    grids, aggregate, split, switch_decisions = {}, {}, {}, {}
    for topology in topology_ids:
        layout = LAYOUTS[topology]
        lags = tuple(range(0, 3 * len(layout.sites), 3))
        cases, fits = [], []
        grid = dict(state_buses=layout.sites, pcc_buses=layout.controls,
                    monitor_and_equal_split_buses=layout.monitors, day_lags=lags, cases=cases, fits=fits)
        grids[topology] = grid
        last_profile = None
        for cond in CONDITIONS:
            profile = (cond.load, cond.pv, cond.slack)
            if profile != last_profile:
                feeder = make_feeder(topology, slack=cond.slack, base_load_scale=cond.load)
                grid["topology"] = feeder.topology_metadata
                powers = construct_profiles(gross, solar, feeder, sites=layout.sites,
                                            day_lags=lags, solar_scale=cond.pv)
                train, heldout = split_indices(len(powers), day_lags=lags, test_days=test_days)
                t0 = perf_counter()
                model = VoltageController.fit(
                    feeder, Scenario(cond.name, observed=tuple(layout.sites.index(b) for b in layout.meters)),
                    powers[train], train, site_buses=layout.sites,
                    control_buses=layout.controls, monitor_buses=layout.monitors,
                    harmonics=harmonics, seed=seed)
                archive = model.qualification
                fits.append(dict(load_scale=cond.load, pv_scale=cond.pv, slack=cond.slack, train_intervals=len(train),
                                 qualification_intervals=len(archive.buckets) if archive else 0,
                                 qualified_archive_commands=len(archive.actions) if archive else 0,
                                 fit_seconds=perf_counter() - t0))
                last_profile = profile
                split = dict(train_first=str(times[train[0]]), train_last=str(times[train[-1]]),
                             last_raw_training_timestamp=str(times[train[-1] + 48 * max(lags)]),
                             test_first=str(times[heldout[0]]), test_last=str(times[heldout[-1]]),
                             last_raw_test_timestamp=str(times[heldout[-1] + 48 * max(lags)]),
                             test_intervals_per_case=len(heldout))
            case, differences, dispatch = evaluate_case(
                feeder, model, layout, heldout, powers[heldout], cond, seed)
            cases.append(case)
            if layout is IEEE33_LAYOUT:
                switch_decisions[topology, cond.name] = dispatch
            for baseline, deltas in differences.items():
                for name, delta in deltas.items():
                    aggregate.setdefault((baseline, name), []).append(delta)
            if progress:
                print(f"{topology} {cond.name}: {case['unique_contexts']} contexts", flush=True)
    # Cases use the same dates and household traces. Average them *before* resampling days.
    macro = {m: {} for m in METHODS[:-1]}
    for (baseline, metric), blocks in aggregate.items():
        difference = np.mean(blocks, axis=0)
        effect = paired_effect(difference, heldout, seed)
        effect["moving_block_ci95_by_days"] = {
            length: moving_block_interval(difference, heldout, length, seed)
            for length in (7, 34) if test_days > length}
        macro[baseline][metric] = effect
    report = dict(
        seed=seed, harmonics=harmonics, voltage_tick_pu=.0001, monitored_deadband=DEADBAND,
        all_bus_limits_pu=[.95, 1.05], fleet_p_and_abs_q_budgets_pu=(STEP * np.asarray(BUDGET)).tolist(),
        split=split, topologies=grids, cases=len(topology_ids) * len(CONDITIONS),
        macro_policy_metrics=summarize_policies(grids), macro_paired_improvement_over=macro,
        protocol=[
            "One measured Ausgrid household, lagged and scaled into synthetic nodal PQ series.",
            "Each known load/PV/slack configuration is fit only on the early training window. "
            "Live derating and meter outage reuse the nominal fit.",
            "In outage_derating_recovery the half-hour index modulo 336 determines both temporary events. "
            "Reports are unavailable in [116,164); the last PCC loses positive Q in [132,204). "
            "The original reports and capability return at the end of their respective windows.",
            "Solar and slack stresses were chosen using uncontrolled AC to exercise both export and "
            "overvoltage on all four topologies. Conditions are equally weighted stress tests.",
            "All policies receive the same clock, category reports, local capability and fleet budgets. "
            "Legacy and adaptive share the complete station action space; equal split has its legacy preset subset.",
            "Metrics use nonlinear AC on the same continuous held-out PQ intervals. Head export uses "
            "negative real slack power from branch currents, including losses, before dispatch.",
            "Each case and each half-hour have equal weight. Paired differences are comparator minus adaptive. "
            "95% intervals resample entire days 500 times and retain pairing across all cases. "
            "Macro sensitivity intervals sample circular contiguous 7/34-day blocks to allow serial dependence "
            "from weather and the 33-day profile lags.",
            "Wall times include context preparation and a local-transport solve per distinct context. "
            "Bytes count serialized neighbor messages, excluding sensor uplink and command delivery.",
        ])
    if all((name, CONDITIONS[0].name) in switch_decisions for name in LAYOUTS
           if LAYOUTS[name] is IEEE33_LAYOUT):
        from switch_replay import evaluate_switches
        report["switch_replay"] = evaluate_switches(switch_decisions, CONDITIONS, gross, solar, heldout, seed)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", nargs="+", choices=tuple(LAYOUTS), default=list(LAYOUTS))
    parser.add_argument("--test-days", type=int, default=115)
    parser.add_argument("--harmonics", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_suite(topology_ids=args.topology, test_days=args.test_days, harmonics=args.harmonics,
                       seed=args.seed, progress=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
