"""PCC count scaling with identical discrete budgets and exact discrete references."""
import argparse
from itertools import combinations
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from distributed import Station, solve_dispatch
from feeder33 import Feeder33


# Prefixes add control sites on both the main feeder and the laterals.
BUSES = (14, 18, 33, 6, 22, 25, 9, 11, 16, 20, 28, 31, 3, 5, 8, 12, 24, 30)
MONITORS = (18, 33)
SCENARIO_LOAD = np.array([.70, 1., 1.30])
SCENARIO_PROBABILITY = np.array([.20, .50, .30])
Q_STEP = .0125
DEADBAND = .020
BUDGET = 3


def options_for(shifts):
    """Enumerate only feasible active subsets; no unrestricted Cartesian array."""
    n = len(shifts)
    subsets = [s for k in range(min(BUDGET, n) + 1) for s in combinations(range(n), k)]
    incidence = np.zeros((len(subsets), n), dtype=bool)
    for row, group in enumerate(subsets):
        incidence[row, list(group)] = True
    return incidence, incidence @ shifts


def ac_losses(feeder, loads, buses, schedules):
    """Physical reference, including the option of no control; chunk by schedule."""
    monitor_indices = np.asarray(MONITORS) - 1
    indices = np.asarray(buses) - 1
    losses = np.empty(len(schedules))
    for start in range(0, len(schedules), 64):
        group = schedules[start:start+64]
        cases = np.broadcast_to(loads, (len(group),) + loads.shape).copy()
        cases[:, :, indices, 1] -= group[:, None, :]*Q_STEP
        voltage = np.abs(feeder.ac_batch(cases.reshape(-1, 33, 2))[:, monitor_indices])
        deviation = voltage.reshape(len(group), len(loads), len(MONITORS)) - 1.
        losses[start:start+len(group)] = (
            np.maximum(np.abs(deviation)-DEADBAND, 0).mean(axis=2) @ SCENARIO_PROBABILITY)
    return losses


def median_seconds(function, repeats):
    function()
    times = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        times.append(perf_counter()-start)
    return median(times)


def experiment(counts=(3, 6, 12, 18), *, tick=.0001, repeats=5):
    counts = tuple(counts)
    if not counts or min(counts) < 1 or max(counts) > len(BUSES):
        raise ValueError(f'PCC count must be between 1 and {len(BUSES)}')
    if repeats < 1 or not np.isfinite(tick) or tick <= 0:
        raise ValueError('repeats and tick must be positive')
    feeder = Feeder33()
    reference = feeder.nominal_pq
    loads = SCENARIO_LOAD[:, None, None] * reference
    monitor_indices = np.asarray(MONITORS) - 1
    base_v = np.abs(feeder.ac_batch(loads))[:, monitor_indices]
    # The distribution is explicit and fixed to isolate controller-count growth.
    def risk(shifts):
        deviation = (base_v - 1.)[None] + shifts[:, None, :]
        return np.maximum(np.abs(deviation) - DEADBAND, 0.).mean(axis=2) @ SCENARIO_PROBABILITY

    results = []
    for count in counts:
        buses = BUSES[:count]
        perturbed = np.repeat(reference[None], count, axis=0)
        perturbed[np.arange(count), np.asarray(buses) - 1, 1] -= Q_STEP
        # Middle load state is the reference for each station's voltage increments.
        shifts = np.abs(feeder.ac_batch(perturbed))[:, monitor_indices] - base_v[1]
        stations = [Station(bus, [(0., 0.), (0., Q_STEP)], [np.zeros(len(MONITORS)), dv],
                            [0., 0.], resources=[[0], [1]]) for bus, dv in zip(buses, shifts)]
        shift_ticks = np.rint(shifts/tick).astype(int)
        def distributed():
            return solve_dispatch(stations, risk, tick=tick, resource_limits=(BUDGET,))
        def centralized():
            _, candidate_ticks = options_for(shift_ticks)
            return np.min(risk(candidate_ticks*tick))

        schedule, all_ticks = options_for(shift_ticks)
        reference_scores = risk(all_ticks*tick)
        best = float(reference_scores.min())
        selected = distributed()
        physical = ac_losses(feeder, loads, buses, schedule)
        selected_row = np.flatnonzero(np.all(schedule == selected.action_indices, axis=1)).item()
        gap = float(abs(selected.objective-best))
        if gap > 1e-12 or abs(reference_scores[selected_row] - best) > 1e-12:
            raise AssertionError('distributed solution differs from constrained enumeration')
        diagnostics = selected.diagnostics
        states = diagnostics['forward_entries'] + [diagnostics['terminal_entries']]
        results.append(dict(controller_count=count, buses=buses, potential_schedules=2**count,
                            feasible_schedules=len(schedule), reachable_states_per_station=states,
                            terminal_states=states[-1], max_forward_entries=max(states[:-1], default=0),
                            neighbor_messages=diagnostics['messages'], wire_bytes=diagnostics['wire_bytes'],
                            optimal_binned_objective=best, distributed_objective_gap=gap,
                            selected_buses=[b for b, a in zip(buses, selected.action_indices) if a],
                            ac_no_control_loss=float(physical[0]),
                            ac_selected_loss=float(physical[selected_row]),
                            ac_oracle_loss=float(physical.min()),
                            ac_selected_regret=float(physical[selected_row]-physical.min()),
                            ac_validation_snapshots=len(schedule)*len(loads),
                            calibration_ac_snapshots=count+len(loads),
                            distributed_seconds=median_seconds(distributed, repeats),
                            constrained_brute_seconds=median_seconds(centralized, repeats),
                            cached_brute_scoring_seconds=median_seconds(lambda: risk(all_ticks*tick).min(), repeats)))
    return dict(feeder='IEEE 33-bus, nominal base_load_scale=0.65 and slack=1.02',
                monitors=MONITORS, deadband_pu=DEADBAND, forecast='specified load-scenario mixture',
                scenario_load_multipliers=SCENARIO_LOAD.tolist(),
                scenario_weights=SCENARIO_PROBABILITY.tolist(), tick_pu=tick,
                q_step_pu=Q_STEP, max_active_stations=BUDGET, edge_protocol='local JSON neighbor messages',
                timing='warm median, per solve, excludes shared calibration and independent AC validation',
                timing_repeats=repeats, results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', nargs='+', type=int, default=(3, 6, 12, 18))
    parser.add_argument('--tick', type=float, default=.0001)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = json.dumps(experiment(args.counts, tick=args.tick, repeats=args.repeats), indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result+'\n', encoding='utf-8')
    else:
        print(result)
