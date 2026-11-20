"""Out-of-year confirmation on all available 2012–13 Customer 12 intervals."""
import argparse
import csv
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from controller import VoltageController
from coordination_benchmark import Scenario
from measured_profiles import DATA_PATH, construct_profiles, profile_scales, read_ausgrid
from topologies import make_feeder
from topology_benchmark import (CONDITIONS, LAYOUTS, evaluate_case,
                                moving_block_interval, paired_effect, summarize_policies)


EXTERNAL_DATA = DATA_PATH.with_name("ausgrid_customer12_2012_2013_wh.csv")
EXTERNAL_GRIDS = ("ieee33", "case69")
EXTERNAL_CONDITIONS = tuple(c for c in CONDITIONS if c.name in ("nominal", "solar_export"))
TEMPORAL_PROTOCOLS = {
    "blackout": "Primary: nominal feeder with intermittent category blackout.",
    "solar_export": "Secondary: solar-export operating condition with normal category reports.",
    "alternating": "Independent-year replication of a development-year diagnostic: solar export, two meters reporting on alternating half-hours.",
}


def read_daily_meter(path=EXTERNAL_DATA):
    """Read actual Wh per half hour and return average kW at local interval starts."""
    labels = [f"{(k // 2) % 24}:{30 * (k % 2):02d}" for k in range(1, 49)]
    readings = {}
    with open(path, newline="", encoding="ascii") as src:
        rows = csv.reader(src)
        if next(rows, []) != ["date", "channel", *labels]:
            raise ValueError("expected 48 interval-end meter columns from 0:30 to 0:00")
        for day, channel, *values in rows:
            key = (date.fromisoformat(day), channel)
            if channel not in ("GC", "GG") or key in readings or len(values) != 48:
                raise ValueError("invalid or duplicate daily channel")
            wh = np.asarray(values, dtype=int)
            if np.any(wh < 0):
                raise ValueError("negative energy measurement")
            readings[key] = wh
    days = sorted({key[0] for key in readings})
    if not days or len(readings) != 2 * len(days) or (days[-1] - days[0]).days != len(days) - 1:
        raise ValueError("require consecutive dates with both GC and GG")
    times = np.datetime64(days[0], "m") + np.arange(48 * len(days)) * np.timedelta64(30, "m")
    gross, pv = (np.concatenate([readings[day, channel] for day in days]) / 500.
                 for channel in ("GC", "GG"))
    return times, gross, pv


def replay(*, training_path=DATA_PATH, test_path=EXTERNAL_DATA, seed=20261120, harmonics=16,
           temporal=None, progress=False):
    if temporal is not None and temporal not in TEMPORAL_PROTOCOLS:
        raise ValueError("unknown temporal replay protocol")
    train_times, gross, pv = read_ausgrid(training_path)
    test_times, external_gross, external_pv = read_daily_meter(test_path)
    if train_times[-1] >= test_times[0]:
        raise ValueError("external confirmation must begin after the development-year records")
    train = np.arange(160 * 48)
    scale = profile_scales(gross, pv)
    lags = tuple(range(0, 36, 3))
    if len(train_times) < len(train) + max(lags) * 48:
        raise ValueError("training year does not cover the full lagged fitting window")
    if len(test_times) <= (max(lags) + 34) * 48:
        raise ValueError("external year must cover site lags and 34-day uncertainty blocks")
    heldout = np.arange(len(test_times) - max(lags) * 48)
    grids, paired = {}, {}
    for topology in EXTERNAL_GRIDS:
        layout = LAYOUTS[topology]
        cases = []
        conditions = EXTERNAL_CONDITIONS
        if temporal:
            condition_name = "nominal" if temporal == "blackout" else "solar_export"
            conditions = tuple(c for c in conditions if c.name == condition_name)
        for condition in conditions:
            feeder = make_feeder(topology, slack=condition.slack, base_load_scale=condition.load)
            mapping = dict(sites=layout.sites, day_lags=lags, solar_scale=condition.pv,
                           reference_scales=scale)
            # Restrict access to the exact raw support of the original 160-day fit.
            raw_end = len(train) + 48 * max(lags)
            training = construct_profiles(gross[:raw_end], pv[:raw_end], feeder, **mapping)
            scenario = Scenario(condition.name, observed=tuple(layout.sites.index(b) for b in layout.meters))
            model = VoltageController.fit(feeder, scenario, training, train,
                                          site_buses=layout.sites, control_buses=layout.controls,
                                          monitor_buses=layout.monitors, seed=seed, harmonics=harmonics)
            external = construct_profiles(external_gross, external_pv, feeder, **mapping)
            if temporal:
                from temporal_benchmark import evaluate_temporal
                case, differences = evaluate_temporal(feeder, model, layout, heldout, external,
                                                      blackout=temporal == "blackout",
                                                      alternating=temporal == "alternating", seed=seed)
                case.update(condition=condition.name, pv_scale=condition.pv, load_scale=condition.load,
                            slack=condition.slack)
            else:
                case, differences, adaptive = evaluate_case(feeder, model, layout, heldout, external, condition, seed)
                case["adaptive_dispatch_sha256_float64le"] = sha256(np.asarray(adaptive, dtype="<f8").tobytes()).hexdigest()
            cases.append(case)
            for baseline, metrics in differences.items():
                for metric, delta in metrics.items():
                    paired.setdefault((baseline, metric), []).append(delta)
            if progress:
                print(f"{topology} {condition.name}: {len(heldout)} external intervals", flush=True)
        grids[topology] = dict(cases=cases, state_buses=layout.sites, pcc_buses=layout.controls,
                               monitor_buses=layout.monitors)
    effects = {}
    for (baseline, metric), arrays in paired.items():
        delta = np.mean(arrays, axis=0)
        effects.setdefault(baseline, {})[metric] = dict(
            **paired_effect(delta, heldout, seed),
            moving_block_ci95_by_days={d: moving_block_interval(delta, heldout, d, seed) for d in (7, 34)})
    report = dict(
        design="Same household, subsequent measured year. Known grid/operating condition; scenario-scaled nodal PQ.",
        fitting="Each condition uses only the original training window for thresholds, MPS, voltage response and control calibration.",
        scoring="Each case and interval has equal weight. Bootstrap paired differences by date after averaging cases.",
        seed=seed, harmonics=harmonics, site_lags_days=lags, gross_reference_kw=scale[0], pv_reference_kw=scale[1],
        training_intervals=len(train), training_first=str(train_times[0]), training_last=str(train_times[train[-1]]),
        last_raw_training=str(train_times[train[-1] + 48*max(lags)]),
        external_intervals_per_case=len(heldout), external_days_per_case=len(heldout)//48,
        external_first=str(test_times[0]), external_last=str(test_times[heldout[-1]]),
        external_last_raw=str(test_times[-1]), cases=sum(len(x["cases"]) for x in grids.values()),
        topologies=grids)
    if temporal:
        report["mode"] = TEMPORAL_PROTOCOLS[temporal]
        report["history_initialization"] = "Empty buffer at the first external interval."
        report["macro_memoryless_minus_lag"] = effects
    else:
        report.update(macro_policies=summarize_policies(grids), macro_paired_improvement_over=effects)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporal", nargs="?", const="blackout", choices=tuple(TEMPORAL_PROTOCOLS),
                        help="compare lag-1 and lag-48 to memoryless dispatch under the selected telemetry protocol")
    args = parser.parse_args()
    result = replay(temporal=args.temporal, progress=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
