"""Fit a radial feeder offline; dispatch PCC setpoints from categorical telemetry."""
import argparse
from dataclasses import asdict, dataclass, replace
import json

import numpy as np

from adaptive_control import build_qualification, calibrate_responses, qualify_dispatch
from coordination_benchmark import (BUDGET, CONTROL_BUSES, SCENARIOS, build_shift_operator,
                                    context_risk, feasible_equal_splits, reference_magnitude_table,
                                    station_tables)
from distributed import solve_dispatch
from fast_fourier import FactorizedFourier
from feeder33 import action_injections
from measured_profiles import (DATA_PATH, SITES, construct_profiles, fit_categories, read_ausgrid,
                               regime, split_indices, state_power)
from powerflow import ACTIONS
from probability_tt import fit_mixture_em
from realistic_benchmark import DEADBAND, MONITORED, calibrate_multi_terminal
from temporal_control import TemporalModel, category_marginals, with_history


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
    site_buses: tuple
    monitor_buses: tuple
    split_buses: tuple
    responses: dict
    qualification: object = None
    temporal: TemporalModel | None = None

    @property
    def pcc_buses(self):
        return tuple(s.bus for s in self.stations)

    @classmethod
    def fit(cls, feeder, scenario, historical_powers, historical_half_hours,
            *, harmonics=16, tick=.0001, seed=20261120, site_buses=SITES,
            control_buses=CONTROL_BUSES, monitor_buses=MONITORED, split_buses=None,
            calibration_samples=32):
        """Use historical data only; half-hour indices determine the clock regime."""
        if not np.isfinite(tick) or tick <= 0:
            raise ValueError("voltage tick must be finite and positive")
        buckets = regime(historical_half_hours)
        if len(buckets) != len(historical_powers):
            raise ValueError("require one clock index per training snapshot")
        historical_powers = np.asarray(historical_powers, dtype=float)
        site_buses, control_buses, monitor_buses = map(tuple, (site_buses, control_buses, monitor_buses))
        split_buses = tuple(monitor_buses if split_buses is None else split_buses)
        for group in (site_buses, control_buses, monitor_buses, split_buses):
            if (not group or len(set(group)) != len(group)
                    or any(not isinstance(b, (int, np.integer)) or b < 2 or b > len(feeder.nominal_pq)
                           for b in group)):
                raise ValueError("bus groups must contain unique non-slack physical bus IDs")
        if len(split_buses) != 2 or not set(split_buses).issubset(control_buses):
            raise ValueError("the two equal-split buses must be controllable")
        if set(buckets) != set(range(4)):
            raise ValueError("require training records in all four clock buckets")
        thresholds, centers, states = fit_categories(historical_powers, site_buses)
        mixtures = [fit_mixture_em(states[buckets == i], components=3, seed=seed+i,
                                   restarts=2, iterations=85) for i in range(4)]
        constants, contributions, _ = calibrate_multi_terminal(
            feeder, centers, site_buses, monitor_buses,
            dispatch_map=lambda a: action_injections(a, split_buses))
        fixed = FactorizedFourier(constants, contributions, DEADBAND, harmonics)
        middle = state_power(feeder.nominal_pq, centers, site_buses, np.ones(len(site_buses), int))
        stations = tuple(station_tables(feeder, middle, scenario, control_buses=control_buses))
        no_action = ACTIONS.index((0., 0.))
        shift = build_shift_operator(constants[no_action], contributions[no_action],
                                     stations, tick, harmonics, monitored=monitor_buses)
        references = reference_magnitude_table(feeder, historical_powers, reference_buses=split_buses)
        responses = calibrate_responses(feeder, historical_powers, stations, split_buses, monitor_buses,
                                        references, constants, contributions, harmonics, tick,
                                        samples=calibration_samples)
        model = cls(thresholds, states, buckets, references, mixtures, fixed, shift, stations, tick,
                    site_buses, monitor_buses, split_buses, responses)
        model.qualification = build_qualification(model, feeder, historical_powers,
                                                  historical_half_hours, scenario.observed)
        model.temporal = TemporalModel.fit(states, buckets, historical_half_hours, mixtures)
        return model

    def _inputs(self, clock_bucket, observed_categories, q_limits):
        if clock_bucket not in range(4):
            raise ValueError("clock bucket must be in 0..3")
        evidence = {}
        for bus, level in observed_categories.items():
            if bus not in self.site_buses or not isinstance(level, (int, np.integer)) or level not in range(3):
                raise ValueError("evidence requires a configured bus and category 0, 1 or 2")
            evidence[self.site_buses.index(bus)] = int(level)
        q_limits = {} if q_limits is None else q_limits
        if not set(q_limits).issubset(s.bus for s in self.stations):
            raise ValueError("unknown PCC in reactive limits")
        stations = []
        action_rows = []
        for station in self.stations:
            if station.bus not in q_limits:
                stations.append(station)
                action_rows.append(np.arange(len(station.actions)))
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
                                    resources=np.asarray(station.resources)[valid],
                                    available=None if station.available is None else np.asarray(station.available)[valid]))
            action_rows.append(np.flatnonzero(valid))
        return stations, action_rows, evidence

    def _reference(self, stations, clock_bucket, evidence):
        original, choices, _ = feasible_equal_splits(stations, self.tick, reference_buses=self.split_buses)
        mixture = self.mixtures[clock_bucket]
        scores = self.fixed_operator.scores(mixture["weights"], mixture["local_probabilities"], evidence)
        index = np.argmin(scores[original].mean(axis=1))
        return int(original[index]), choices[index]

    def reference_dispatch(self, clock_bucket, observed_categories, *, q_limits=None):
        """Return the feasible six-preset reference's (P,Q) in control-bus order."""
        stations, _, evidence = self._inputs(clock_bucket, observed_categories, q_limits)
        _, choice = self._reference(stations, clock_bucket, evidence)
        return tuple(tuple(map(float, s.actions[i])) for s, i in zip(stations, choice, strict=True))

    def predict_categories(self, clock_bucket, observed_categories, *, history=None, half_hour=None):
        """Conditional predictive probabilities at all sites, in configured site order."""
        _, _, evidence = self._inputs(clock_bucket, observed_categories, None)
        mixture, _ = self.temporal.infer(self.mixtures, self.site_buses, clock_bucket, history, half_hour)
        return category_marginals(mixture, evidence)

    def dispatch(self, clock_bucket, observed_categories, *, q_limits=None, transport="local",
                 method="adaptive", history=None, half_hour=None):
        """Return station P,Q choices. Keys are physical bus IDs. No online AC solves."""
        stations, _, evidence = self._inputs(clock_bucket, observed_categories, q_limits)
        if method == "temporal":
            adjusted = with_history(self, clock_bucket, evidence, history, half_hour)
            return adjusted.dispatch(clock_bucket, observed_categories, q_limits=q_limits,
                                     transport=transport, method="adaptive")
        if history:
            raise ValueError("lagged telemetry requires method='temporal'")
        if method not in ("adaptive", "legacy"):
            raise ValueError("method must be adaptive, legacy or temporal")
        context = context_risk(stations, self.fixed_operator, self.shift_operator,
                               self.mixtures[clock_bucket], evidence, self.training_states,
                               self.reference_voltages, self.training_buckets, clock_bucket, self.tick,
                               monitored=self.monitor_buses, reference_buses=self.split_buses)
        incumbent = solve_dispatch(stations, context.risk, tick=self.tick, resource_limits=BUDGET,
                                   transport=transport)
        if method == "legacy":
            return incumbent
        return qualify_dispatch(self, incumbent, stations, clock_bucket, evidence, transport)


def main():
    from topologies import LAYOUTS, make_feeder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topology', choices=tuple(LAYOUTS), default='ieee33')
    parser.add_argument('--scenario', choices=[s.name for s in SCENARIOS], default='original')
    parser.add_argument('--clock-bucket', type=int, choices=range(4), required=True)
    parser.add_argument('--half-hour', type=int,
                        help='current half-hour index for exact lag clock matching in temporal mode')
    parser.add_argument('--observe', nargs='*', default=[], metavar='BUS=LEVEL')
    parser.add_argument('--observe-30m-ago', nargs='*', default=[], metavar='BUS=LEVEL')
    parser.add_argument('--observe-24h-ago', nargs='*', default=[], metavar='BUS=LEVEL')
    parser.add_argument('--q-limit', nargs='*', default=[], metavar='BUS=MIN,MAX')
    parser.add_argument('--transport', choices=('local', 'process'), default='local')
    parser.add_argument('--method', choices=('adaptive', 'legacy', 'temporal'), default='adaptive')
    args = parser.parse_args()
    try:
        def reports(entries):
            return {int(bus): int(cat) for bus, cat in (s.split('=') for s in entries)}
        observed = reports(args.observe)
        history = {lag: reports(entries) for lag, entries in
                   ((1, args.observe_30m_ago), (48, args.observe_24h_ago)) if entries}
        limits = {int(bus): tuple(map(float, bounds.split(',')))
                  for bus, bounds in (s.split('=') for s in args.q_limit)}
    except ValueError as exc:
        parser.error(str(exc))
    if (history or args.half_hour is not None) and args.method != 'temporal':
        parser.error('historical reports require --method temporal')
    if args.half_hour is not None and (args.half_hour % 48)//12 != args.clock_bucket:
        parser.error('--half-hour must belong to --clock-bucket')
    scenario = next(s for s in SCENARIOS if s.name == args.scenario)
    layout = LAYOUTS[args.topology]
    feeder = make_feeder(args.topology, slack=scenario.slack, base_load_scale=scenario.load,
                         trunk_length_factor=scenario.length)
    _, load, pv = read_ausgrid(DATA_PATH)
    lags = tuple(range(0, 3*len(layout.sites), 3))
    powers = construct_profiles(load, pv, feeder, sites=layout.sites, day_lags=lags, solar_scale=scenario.pv)
    train, _ = split_indices(len(powers), day_lags=lags)
    model = VoltageController.fit(feeder, scenario, powers[train], train, site_buses=layout.sites,
                                  control_buses=layout.controls, monitor_buses=layout.monitors)
    options = {"history": history, "half_hour": args.half_hour} if args.method == "temporal" else {}
    result = model.dispatch(args.clock_bucket, observed, q_limits=limits, transport=args.transport,
                            method=args.method, **options)
    output = asdict(result)
    output['topology'] = args.topology
    output['voltage_monitor_buses'] = layout.monitors
    output['bus_setpoints'] = {s.bus: pq for s, pq in zip(model.stations, result.actions, strict=True)}
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
