import itertools
import unittest

import numpy as np

from distributed import Station, StationAgent, solve_dispatch


def support_stations(remote_available):
    actions = ((0.0, 0.0), (0.0, 0.01))
    return [Station(bus, actions, ((0.,), (.001,)), (0., cost), (True, allowed))
            for bus, cost, allowed in [(3, .35, True), (6, .20, True),
                                       (9, .10, remote_available)]]


def deficiency(shifts):
    return .6 * np.maximum(1. - shifts[:, 0]/.001, 0)**2


class TestDistributedDispatch(unittest.TestCase):
    def test_capacity_change_reallocates_to_node_six(self):
        normal = solve_dispatch(support_stations(True), deficiency)
        capped = solve_dispatch(support_stations(False), deficiency)
        self.assertEqual(normal.action_indices, (0, 0, 1))
        self.assertEqual(capped.action_indices, (0, 1, 0))
        self.assertAlmostEqual(normal.objective, .10)
        self.assertAlmostEqual(capped.objective, .20)
        self.assertEqual(normal.diagnostics['messages'], 4)
        self.assertEqual(normal.diagnostics['forward_entries'], [2, 3])

    def test_multiprocess_neighbors_match_local_wire_and_actions(self):
        stations = support_stations(False)
        local = solve_dispatch(stations, deficiency)
        proc = solve_dispatch(stations, deficiency, transport='process')
        self.assertEqual(proc.actions, local.actions)
        self.assertEqual(proc.objective, local.objective)
        self.assertEqual(proc.diagnostics['wire_bytes'], local.diagnostics['wire_bytes'])

    def test_exact_for_two_monitors_and_binding_resource_limit(self):
        tick = .0007
        stations = [
            Station(i, [(0., 0.), (.01, .01), (.01, -.01)],
                    [[0., 0.], [i*.0012, .001], [-.001, i*.0011]],
                    [0., .00001, .00002], resources=[[0, 0], [1, 1], [1, 2]])
            for i in (1, 2, 3)]
        def risk(shifts):
            return np.abs(shifts - [.006, .009]).sum(axis=1)
        exact = np.inf
        for actions in itertools.product(range(3), repeat=3):
            resource = np.sum([s.resources[i] for s, i in zip(stations, actions)], axis=0)
            if np.any(resource > [2, 3]):
                continue
            rounded = np.sum([np.rint(np.array(s.voltage_shifts[i])/tick)
                              for s, i in zip(stations, actions)], axis=0)*tick
            exact = min(exact, sum(s.costs[i] for s, i in zip(stations, actions))
                        + risk(rounded[None, :])[0])
        actual = solve_dispatch(stations, risk, tick=tick, resource_limits=(2, 3))
        self.assertAlmostEqual(exact, actual.objective, places=14)
        self.assertTrue(np.all(np.array(actual.diagnostics['resources_used']) <= [2, 3]))
        physical = np.sum([s.voltage_shifts[i] for s, i in zip(stations, actual.action_indices)], axis=0)
        self.assertLessEqual(np.max(np.abs(physical - actual.voltage_shift)), len(stations)*tick/2)

    def test_infeasible_is_an_error(self):
        s = Station(3, [(0,0)], [[0]], [0], resources=[[2]])
        with self.assertRaisesRegex(ValueError, 'feasible'):
            solve_dispatch([s], deficiency, resource_limits=[1])
        with self.assertRaisesRegex(ValueError, 'feasible'):
            solve_dispatch(support_stations(True), lambda shifts: np.full(len(shifts), np.inf))

    def test_intermediate_station_propagates_local_infeasibility(self):
        stations = support_stations(True)
        s = stations[1]
        stations[1] = Station(s.bus, s.actions, s.voltage_shifts, s.costs, (False, False))
        for mode in ('local', 'process'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'feasible'):
                solve_dispatch(stations, deficiency, transport=mode, timeout=15)

    def test_wire_excludes_action_tables_and_preserves_ties(self):
        station = Station(6, [(0, 0), (0, .1)], [[0], [0]], [0, 0])
        agent = StationAgent(station, .01, (), 1)
        agent.advance()
        self.assertEqual(agent.offer(), b'[[[0],0.0]]')
        self.assertEqual(solve_dispatch([station], deficiency).action_indices, (0,))


if __name__ == '__main__':
    unittest.main()
