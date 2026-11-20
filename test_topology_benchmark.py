"""Independent checks for held-out voltage, head flow and paired policy replay."""
from types import SimpleNamespace
import unittest

import numpy as np

from controller import VoltageController
from coordination_benchmark import Scenario
from distributed import Station
from measured_profiles import construct_profiles, read_ausgrid
from topology_benchmark import (CONDITIONS, IEEE33_LAYOUT, LAYOUTS, METHODS, capacity_oracle, dispatch_intervals,
                                evaluate_case, head_active_power, moving_block_interval, paired_effect,
                                validate_actions, voltage_scores)
from topologies import make_feeder


class TopologyBenchmarkTests(unittest.TestCase):
    def test_capacity_oracle_separates_avoidable_from_infeasible(self):
        loads = np.zeros((48, 2, 2))
        loads[:, 1, 0] = np.tile([.49, .515, .6], 16)
        feeder = SimpleNamespace(ac_batch=lambda x: 1. - .1 * x[:, :, 0])
        station = Station(2, [(0., 0.), (.025, 0.)], [[0.], [.0025]], [0., 0.],
                          resources=[[0, 0], [1, 0]])
        dispatch = {"no_action": np.zeros((48, 1, 2)),
                    "full": np.tile([.025, 0.], (48, 1, 1))}
        oracle = capacity_oracle(feeder, SimpleNamespace(stations=[station], tick=.0001),
                                 loads, np.arange(48), dispatch)
        self.assertEqual(oracle["feasible_joint_commands"], 2)
        self.assertEqual(oracle["counts"]["no_action"]["unsafe_but_feasible"], 16)
        self.assertEqual(oracle["counts"]["full"]["unsafe_but_feasible"], 0)
        self.assertEqual(oracle["counts"]["full"]["unsafe_for_every_feasible_action"], 16)

    def test_head_power_accounts_for_all_feeder_losses(self):
        for name in ("ieee33_tie_18_33", "case69"):
            with self.subTest(name=name):
                feeder = make_feeder(name)
                powers = np.stack([feeder.nominal_pq, -.5 * feeder.nominal_pq])
                v = feeder.ac_batch(powers)
                parents = feeder.parent[1:]
                z = feeder.impedance[1:]
                currents = (v[:, parents] - v[:, 1:]) / z
                losses = (abs(currents) ** 2 * z.real).sum(axis=1)
                measured = head_active_power(feeder, v)
                np.testing.assert_allclose(measured, powers[:, :, 0].sum(axis=1) + losses,
                                           rtol=1e-7, atol=1e-9)
                self.assertGreater(measured[0], 0.)
                self.assertLess(measured[1], 0.)

    def test_monitored_deadband_and_full_network_limits(self):
        v = np.array([[1., .94, 1.], [1., 1.04, 1.051]], complex)
        metrics, losses, unsafe = voltage_scores(v, (2,))
        np.testing.assert_allclose(losses, [.04, .02])
        np.testing.assert_array_equal(unsafe, [1, 1])
        self.assertEqual(metrics["all_bus_overvoltage_fraction"], .5)
        self.assertEqual(metrics["all_bus_undervoltage_fraction"], .5)
        self.assertAlmostEqual(metrics["worst_all_bus_limit_excess"], .01)

    def test_outage_and_derating_visible_to_every_policy(self):
        layout = IEEE33_LAYOUT
        powers = np.zeros((3, 33, 2))
        powers[1, :, 0] = 2.
        calls = []
        actions = ((0., 0.),) * 3

        def reference(bucket, observations, *, q_limits):
            calls.append(("equal_split", bucket, observations, q_limits))
            return actions

        def dispatch(bucket, observations, *, q_limits, method):
            calls.append((method, bucket, observations, q_limits))
            return SimpleNamespace(actions=actions, diagnostics={"messages": 4, "wire_bytes": 13})

        model = SimpleNamespace(
            thresholds=np.tile([.3, .7], (len(layout.sites), 1)),
            stations=[Station(bus, [(0., 0.)], [[0.]], [0.]) for bus in layout.controls],
            dispatch=dispatch, reference_dispatch=reference)
        for condition in (c for c in CONDITIONS if c.derated or c.meter_outage):
            calls.clear()
            _, diagnostic, meta = dispatch_intervals(model, layout, np.array([0, 1, 12]), powers, condition)
            masks = {layout.controls[-1]: (-.05, 0.)} if condition.derated else {}
            self.assertEqual(meta["unique_contexts"], 3 if condition.derated else 2)
            for i in range(0, len(calls), len(METHODS)):
                group = calls[i:i+len(METHODS)]
                self.assertEqual(tuple(item[0] for item in group), METHODS)
                self.assertTrue(all(item[1:] == group[0][1:] for item in group))
                self.assertEqual(group[0][-1], masks)
                self.assertEqual(tuple(group[0][2]), () if condition.meter_outage else layout.meters)
            self.assertEqual(diagnostic["legacy"]["neighbor_messages_per_unique_context"], 4)
        infeasible = np.array([[(.025, 0.)] * 3])
        with self.assertRaises(AssertionError):
            validate_actions(infeasible, model, {})

    def test_paired_day_resampling_keeps_within_day_dependence(self):
        values = np.repeat([-1., 3.], 48)
        result = paired_effect(values, np.arange(96), 7)
        self.assertEqual(result["mean"], 1.)
        self.assertEqual(result["day_block_ci95"], (-1., 3.))
        self.assertEqual(moving_block_interval(values, np.arange(96), 2, 7), [1., 1.])
        with self.assertRaises(ValueError):
            moving_block_interval(values[:-1], np.arange(95), 1, 7)

    def test_temporary_event_overlap_and_recovery_in_chronological_order(self):
        layout = IEEE33_LAYOUT
        indices = np.array([115, 116, 131, 132, 163, 164, 203, 204, 335, 336 + 116])
        powers = np.zeros((len(indices), 33, 2))
        stations = [Station(bus, [(0., 0.), (0., .025)], [[0.], [.001]], [0., 0.])
                    for bus in layout.controls]
        def action(bucket, observed, *, q_limits):
            return ((0., .025 if observed else 0.), (0., 0.), (0., 0. if q_limits else .025))

        def dispatch(bucket, observed, *, q_limits, method):
            return SimpleNamespace(actions=action(bucket, observed, q_limits=q_limits),
                                   diagnostics={"messages": 4, "wire_bytes": 60})

        model = SimpleNamespace(thresholds=np.tile([.3, .7], (len(layout.sites), 1)),
                                stations=stations, reference_dispatch=action, dispatch=dispatch)
        condition = next(c for c in CONDITIONS if c.time_varying)
        chosen, _, evidence = dispatch_intervals(model, layout, indices, powers, condition)
        self.assertEqual(len(evidence["operating_modes"]), 4)
        for table in chosen.values():
            np.testing.assert_array_equal(table[:, 0, 1] > 0, [1,0,0,0,0,1,1,1,1,0])
            np.testing.assert_array_equal(table[:, 2, 1] > 0, [1,1,1,0,0,0,0,1,1,1])

    def test_case69_replay_recomputes_ac_on_the_same_continuous_loads(self):
        f = make_feeder("case69")
        layout = LAYOUTS["case69"]
        _, gc, pv = read_ausgrid()
        powers = construct_profiles(gc, pv, f, sites=layout.sites, day_lags=tuple(range(0, 36, 3)))
        train = np.arange(10 * 48)
        test = 215 * 48 + np.arange(0, 48, 6)
        model = VoltageController.fit(f, Scenario("test"), powers[train], train, site_buses=layout.sites,
                                      monitor_buses=layout.monitors, control_buses=layout.controls,
                                      harmonics=4, calibration_samples=3)
        snapshots = powers[test].copy()
        case, deltas, actions = evaluate_case(f, model, layout, test, snapshots, CONDITIONS[0], seed=7)
        np.testing.assert_array_equal(snapshots, powers[test])
        adjusted = snapshots.copy()
        adjusted[:, np.asarray(layout.controls)-1] -= actions
        losses = voltage_scores(f.ac_batch(adjusted), layout.monitors)[1]
        self.assertAlmostEqual(losses.mean(), case["policies"]["adaptive"]["mean_monitor_loss"])
        for baseline in METHODS[:-1]:
            self.assertAlmostEqual(deltas[baseline]["mean_monitor_loss"].mean() + losses.mean(),
                                   case["policies"][baseline]["mean_monitor_loss"])


if __name__ == "__main__":
    unittest.main()
