"""Matched-clock training-day placebo for lagged reports in high-PV operation."""
import argparse
import json
from pathlib import Path

import numpy as np

from controller import VoltageController
from coordination_benchmark import Scenario
from measured_profiles import DATA_PATH, apply_categories, construct_profiles, read_ausgrid, regime, split_indices
from temporal_benchmark import available_reports, effects_with_blocks, evidence, lag_reports
from topologies import LAYOUTS, make_feeder
from topology_benchmark import CONDITIONS, validate_actions, voltage_scores


def donor_rows(train, heldout, max_site_lag, seed):
    """One sampled complete training day per replay day, shared by all meters.

    Return the original training timestamps in heldout clock order. The
    first replay day is mapped too, but nonexistent pre-replay reports are
    always masked later, equally for true and placebo histories.
    """
    train, heldout = np.asarray(train), np.asarray(heldout)
    if (not len(train) or not len(heldout) or np.any(np.diff(heldout) != 1)
            or heldout[0] % 48 or len(heldout) % 48):
        raise ValueError("replay requires consecutive complete held-out days and training records")
    if len(np.unique(train)) != len(train):
        raise ValueError("training timestamps must be unique")
    days, counts = np.unique(train//48, return_counts=True)
    days = days[(counts == 48) & ((days+1)*48-1 + max_site_lag*48 < heldout[0])]
    if not len(days):
        raise ValueError("no complete donor day with disjoint raw source support")
    draw = np.random.default_rng(seed).choice(days, len(heldout)//48,
                                              replace=len(heldout)//48 > len(days))
    return np.repeat(draw, 48)*48 + heldout % 48


def paired_replay(model, layout, indices, reports, placebo_reports, lag):
    previous = (lag_reports(reports, lag), lag_reports(placebo_reports, lag))
    if np.any((previous[0] < 0) != (previous[1] < 0)):
        raise ValueError("history availability masks must match")
    clock = regime(indices)
    keys = [np.column_stack((clock, np.where(np.any(rows >= 0, axis=1), regime(indices-lag), -1),
                             reports, rows)) for rows in previous]
    _, first, inv = np.unique(np.concatenate(keys), axis=0, return_index=True, return_inverse=True)
    meters = [layout.sites.index(b) for b in layout.meters]
    decisions, probabilities = [], []
    for position in first:
        variant, i = divmod(int(position), len(indices))
        past = evidence(layout.meters, previous[variant][i])
        current = evidence(layout.meters, reports[i])
        args = dict(history={lag: past}, half_hour=int(indices[i])) if past else {}
        dispatch = model.dispatch(int(clock[i]), current, method="temporal" if past else "adaptive", **args)
        decisions.append(dispatch.actions)
        # Held-out current readings never enter a pre-report category prediction.
        probabilities.append(model.predict_categories(int(clock[i]), {}, **args)[meters])
    actions = np.asarray(decisions)
    validate_actions(actions, model, {})
    if not np.allclose(np.sum(probabilities, axis=2), 1.) or np.any(np.asarray(probabilities) <= 0.):
        raise ValueError("invalid pre-report probabilities")
    expanded = inv.reshape(2, len(indices))
    return actions[expanded], np.asarray(probabilities)[expanded], len(first)


def evaluate(feeder, model, layout, indices, powers, reports, donors, lag, seed):
    actions, probabilities, count = paired_replay(model, layout, indices, reports, donors, lag)
    metrics, paired = {}, {}
    scores = []
    for i, name in enumerate(("true", "placebo")):
        adjusted = powers.copy()
        adjusted[:, np.asarray(layout.controls)-1] -= actions[i]
        voltage, loss, unsafe = voltage_scores(feeder.ac_batch(adjusted), layout.monitors)
        observed = np.take_along_axis(probabilities[i], reports[:, :, None], axis=2)[:, :, 0]
        nll = -np.log(observed).mean(axis=1)
        metrics[name] = dict(**voltage, pre_report_meter_nll=float(nll.mean()))
        scores.append(dict(mean_monitor_loss=loss, all_bus_unsafe_fraction=unsafe, pre_report_meter_nll=nll))
    changed = np.any(actions[0] != actions[1], axis=(1, 2))
    if np.any(changed[:lag]):
        raise AssertionError("identical empty histories yielded different dispatch")
    deltas = {k: scores[0][k]-scores[1][k] for k in scores[0]}
    for k, delta in deltas.items():
        paired[k] = effects_with_blocks(delta, indices, seed)
    return dict(metrics=metrics, true_minus_placebo=paired, changed_dispatch_intervals=int(changed.sum()),
                shared_unique_contexts=count, lag_half_hours=lag, intervals=len(indices),
                received_history_intervals=max(0, len(indices)-lag)), deltas


def run(*, test_days=115, seed=20261120, progress=False):
    times, load, pv = read_ausgrid(DATA_PATH)
    scenario = next(c for c in CONDITIONS if c.name == "solar_export")
    cases, aggregate = {}, {}
    for topology in ('ieee33', 'case69'):
        layout = LAYOUTS[topology]
        feeder = make_feeder(topology, slack=scenario.slack, base_load_scale=scenario.load)
        site_lags = tuple(range(0, 3*len(layout.sites), 3))
        powers = construct_profiles(load, pv, feeder, sites=layout.sites, day_lags=site_lags,
                                    solar_scale=scenario.pv)
        train, test = split_indices(len(powers), day_lags=site_lags, test_days=test_days)
        donor_timestamps = donor_rows(train, test, max(site_lags), seed)
        meters = [layout.sites.index(b) for b in layout.meters]
        model = VoltageController.fit(feeder, Scenario('solar_export', observed=tuple(meters)),
                                      powers[train], train, site_buses=layout.sites,
                                      control_buses=layout.controls, monitor_buses=layout.monitors, seed=seed)
        categories = apply_categories(powers[test], model.thresholds, site_buses=layout.sites)[:, meters]
        reports, _ = available_reports(test, categories, False)
        donor = apply_categories(powers[donor_timestamps], model.thresholds, site_buses=layout.sites)[:, meters]
        donor = np.where(reports < 0, -1, donor)
        cases[topology] = {}
        for name, lag in (('half_hour', 1), ('previous_day', 48)):
            result, delta = evaluate(feeder, model, layout, test, powers[test], reports, donor, lag, seed)
            cases[topology][name] = result
            for key, values in delta.items():
                aggregate.setdefault((name, key), []).append(values)
            if progress:
                print(topology, name, result['shared_unique_contexts'], flush=True)
    macro = {name:{} for name in ('half_hour', 'previous_day')}
    for (name, k), values in aggregate.items():
        macro[name][k] = effects_with_blocks(np.mean(values, axis=0), test, seed)
    return dict(seed=seed, scenario=scenario.name, load_scale=scenario.load, pv_scale=scenario.pv,
                slack=scenario.slack, test_days=test_days, cases=cases, macro_true_minus_placebo=macro,
                donor_training_days=(donor_timestamps[::48]//48).tolist(),
                last_possible_donor_raw_timestamp=str(times[donor_timestamps.max()+48*max(site_lags)]),
                first_test_timestamp=str(times[test[0]]),
                protocol=[
                    "Only historical inputs differ. Current meter categories, fitted temporal policy, "
                    "initially empty buffers, feasible controls, and held-out continuous injections are identical.",
                    "Replace each complete held-out report day by a randomly selected complete training day "
                    "at the same 48 clock slots. Both meters use the same donor, retaining their joint state "
                    "and within-day shape. Lag this delivered stream identically to the real reports.",
                    "Donors are drawn without replacement when enough days exist. No donor's lagged raw "
                    "household measurements overlap the held-out portion. The fixed seed is shared across grids.",
                    "Pre-report meter NLL is in nats per meter. True minus placebo below zero favors real history. "
                    "Average grids at matching timestamps before paired day and 7/34-day moving-block intervals.",
                    "A single donor allocation does not integrate donor-draw uncertainty. Training and test "
                    "seasons can differ, so this tests clock-matched substitution and does not isolate a causal "
                    "effect from all distribution shift."
                ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261120)
    parser.add_argument('--test-days', type=int, choices=range(1, 116), default=115)
    args = parser.parse_args()
    result = run(test_days=args.test_days, seed=args.seed, progress=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
