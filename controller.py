"""Fit the IEEE33 model offline; dispatch PCC setpoints from a current observation."""
import argparse
from dataclasses import asdict, dataclass, replace
import json

import numpy as np

from coordination_benchmark import (BUDGET, SCENARIOS, TIMESHIFTS, build_shift_operator,
                                    context_risk, reference_magnitude_table, station_tables)
from distributed import solve_dispatch
from fast_fourier import FactorizedFourier
from feeder33 import Feeder33
from measured_profiles import (DATA_PATH, SITES, construct_profiles, fit_categories, read_ausgrid,
                               regime, split_indices, state_power)
from powerflow import ACTIONS
from probability_tt import fit_mixture_em
from realistic_benchmark import DEADBAND, calibrate_multi_terminal


@dataclass
class VoltageController:
    thresholds: np.ndarray
    training_states: np.ndarray
    training_buckets: np.ndarray
    reference_voltages: dict
    mixtures: list
    fixed_operator: FactorizedFourier
    shift_operator: FactorizedFourier
    stations: tuple
    tick: float

    @classmethod
    def fit(cls, feeder, scenario, historical_powers, historical_half_hours,
            *, harmonics=16, tick=.0001, seed=20261120):
        """Use historical data only; half-hour indices determine the clock regime."""
        if not np.isfinite(tick) or tick <= 0:
            raise ValueError("voltage tick must be finite and positive")
        buckets = regime(historical_half_hours)
        if len(buckets) != len(historical_powers):
            raise ValueError("require one clock index per training snapshot")
        thresholds, centers, states = fit_categories(historical_powers)
        mixtures = [fit_mixture_em(states[buckets == i], components=3, seed=seed+i,
                                   restarts=2, iterations=85) for i in range(4)]
        constants, contributions, _ = calibrate_multi_terminal(feeder, centers)
        fixed = FactorizedFourier(constants, contributions, DEADBAND, harmonics)
        middle = state_power(feeder.nominal_pq, centers, SITES, np.ones(len(SITES), int))
        stations = tuple(station_tables(feeder, middle, scenario))
        no_action = ACTIONS.index((0., 0.))
        shift = build_shift_operator(constants[no_action], contributions[no_action],
                                     stations, tick, harmonics)
        references = reference_magnitude_table(feeder, historical_powers)
        return cls(thresholds, states, buckets, references, mixtures, fixed, shift, stations, tick)

    def dispatch(self, clock_bucket, observed_categories, *, q_limits=None, transport="local"):
        """Return station P,Q choices. Keys are physical bus IDs. No online AC solves."""
        if clock_bucket not in range(4):
            raise ValueError("clock bucket must be in 0..3")
        evidence = {}
        for bus, level in observed_categories.items():
            if bus not in SITES or not isinstance(level, (int, np.integer)) or level not in range(3):
                raise ValueError("evidence requires a configured bus and category 0, 1 or 2")
            evidence[SITES.index(bus)] = int(level)
        q_limits = {} if q_limits is None else q_limits
        if not set(q_limits).issubset(s.bus for s in self.stations):
            raise ValueError("unknown PCC in reactive limits")
        stations = []
        for station in self.stations:
            if station.bus not in q_limits:
                stations.append(station)
                continue
            low, high = q_limits[station.bus]
            if not np.isfinite([low, high]).all() or low > high:
                raise ValueError("invalid reactive bounds")
            acts = np.asarray(station.actions)
            valid = (acts[:, 1] >= low) & (acts[:, 1] <= high)
            if not valid.any():
                raise ValueError(f"no available reactive setpoint at bus {station.bus}")
            stations.append(replace(station, actions=acts[valid], voltage_shifts=station.voltage_shifts[valid],
                                    costs=np.asarray(station.costs)[valid],
                                    resources=np.asarray(station.resources)[valid]))
        context = context_risk(stations, self.fixed_operator, self.shift_operator,
                               self.mixtures[clock_bucket], evidence, self.training_states,
                               self.reference_voltages, self.training_buckets, clock_bucket, self.tick)
        return solve_dispatch(stations, context.risk, tick=self.tick, resource_limits=BUDGET,
                              transport=transport)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=[s.name for s in SCENARIOS], default='original')
    parser.add_argument('--clock-bucket', type=int, choices=range(4), required=True)
    parser.add_argument('--observe', nargs='*', default=[], metavar='BUS=LEVEL')
    parser.add_argument('--q-limit', nargs='*', default=[], metavar='BUS=MIN,MAX')
    parser.add_argument('--transport', choices=('local', 'process'), default='local')
    args = parser.parse_args()
    try:
        observed = {int(bus): int(cat) for bus, cat in (s.split('=') for s in args.observe)}
        limits = {int(bus): tuple(map(float, bounds.split(',')))
                  for bus, bounds in (s.split('=') for s in args.q_limit)}
    except ValueError as exc:
        parser.error(str(exc))
    scenario = next(s for s in SCENARIOS if s.name == args.scenario)
    feeder = Feeder33(slack=scenario.slack, base_load_scale=scenario.load, trunk_length_factor=scenario.length)
    _, load, pv = read_ausgrid(DATA_PATH)
    powers = construct_profiles(load, pv, feeder, day_lags=TIMESHIFTS, solar_scale=scenario.pv)
    train, _ = split_indices(len(powers), day_lags=TIMESHIFTS)
    model = VoltageController.fit(feeder, scenario, powers[train], train)
    result = model.dispatch(args.clock_bucket, observed, q_limits=limits, transport=args.transport)
    output = asdict(result)
    output['bus_setpoints'] = {s.bus: pq for s, pq in zip(model.stations, result.actions, strict=True)}
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
