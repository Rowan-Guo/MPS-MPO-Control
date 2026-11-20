import unittest

import numpy as np

from coordination_benchmark import (
    BUDGET, CONTROL_BUSES, GUARD_BAND, SCENARIOS, STEP, Scenario,
    empirical_deadband, feasible_equal_splits, paired_safety_guard, reference_actions, reference_envelope,
    risk_functions, station_tables,
)
from fast_fourier import FactorizedFourier
from feeder33 import Feeder33


class TestBenchmarkContracts(unittest.TestCase):
    def test_local_menu_and_six_baselines_respect_shared_budget(self):
        feeder = Feeder33()
        for scenario, expected in [(SCENARIOS[0], 6), (Scenario("derate", reactive_limit_33=True), 4)]:
            stations = station_tables(feeder, feeder.nominal_pq, scenario)
            base, choices, shifts = feasible_equal_splits(stations, 1e-4)
            self.assertEqual(len(base), expected)
            options, all_shifts = reference_actions(stations, 1e-4)
            for choice, shift in zip(choices, shifts, strict=True):
                row, = np.flatnonzero(np.all(options == choice, axis=1))
                np.testing.assert_equal(shift, all_shifts[row])
            fleet = np.stack([np.asarray(s.actions)[options[:, j]] for j, s in enumerate(stations)], axis=1)
            self.assertTrue(np.all(fleet[:, :, 0].sum(axis=1) <= BUDGET[0] * STEP + 1e-12))
            self.assertTrue(np.all(np.abs(fleet[:, :, 1]).sum(axis=1) <= BUDGET[1] * STEP + 1e-12))
            if scenario.reactive_limit_33:
                self.assertTrue(np.all(fleet[:, CONTROL_BUSES.index(33), 1] <= 0))

    def test_lower_and_upper_empirical_tails_match_direct_average(self):
        values = np.array([[.1,.0], [0.,.2], [-.06,.01]])
        shifts = np.array([[-.02, .04], [0., 0.], [.1,-.09]])
        direct = np.maximum(abs(values + shifts[:, None, :]) - GUARD_BAND, 0.).mean(axis=(1,2))
        np.testing.assert_allclose(empirical_deadband(values, GUARD_BAND)(shifts), direct, atol=1e-12)

    def test_envelope_allows_reference_and_rejects_exhausted_voltage_margin(self):
        shift = np.array([.01, -.02])
        reference_voltages = np.array([[.952, 1.0], [.97, 1.047]])
        penalty = reference_envelope(shift, reference_voltages)
        np.testing.assert_equal(penalty(np.array([shift, [.009,-.02], [.01,-.019], [.011,-.021]])),
                                [0., np.inf, np.inf, 0.])

    def test_paired_guard_counts_joint_events_once_and_balances_rescues(self):
        # One control proposal rescues twelve undervoltage snapshots at bus 1,
        # while creating one new overvoltage event simultaneously at buses 2/3.
        tail = np.array([[.948, 1., 1.]] * 12 + [[1., 1.047, 1.047]])
        origin = np.zeros(3)
        proposals = np.array([[0.,0.,0.], [.006,.006,.006], [0.,.006,.006]])
        np.testing.assert_equal(paired_safety_guard(origin, tail)(proposals), [0.,0.,np.inf])
        # With only one rescue, the uncertainty allowance rejects the swap.
        np.testing.assert_equal(paired_safety_guard(origin, tail[[-2,-1]])(proposals), [0.,np.inf,np.inf])

    def test_unseen_measurement_uses_conditional_fourier_prior(self):
        op = FactorizedFourier(np.array([[.01,-.04]]), np.zeros((1,2,2,3)), .02, 4, domain=.2)
        fit = dict(weights=np.ones(1), local_probabilities=np.ones((1,2,3))/3)
        voltages=np.zeros((3,33))
        bucket=np.ones(3,dtype=bool)
        shifts=np.zeros((2,33))
        shifts[1,17] = .02
        _, combined, model, _, _ = risk_functions(op, fit, {0: 1}, voltages, ~bucket, bucket)
        np.testing.assert_array_equal(combined(shifts), model(shifts))


if __name__ == "__main__":
    unittest.main()
