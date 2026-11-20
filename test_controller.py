import unittest
from unittest.mock import patch

import numpy as np

from controller import VoltageController
from coordination_benchmark import Scenario, TIMESHIFTS
from feeder33 import Feeder33
from measured_profiles import DATA_PATH, construct_profiles, read_ausgrid


class TestDispatchEntry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        feeder = Feeder33()
        _, gc, pv = read_ausgrid(DATA_PATH)
        powers = construct_profiles(gc, pv, feeder, day_lags=TIMESHIFTS)
        rows = np.arange(0, 16 * 48, 2)
        cls.model = VoltageController.fit(feeder, Scenario('test'), powers[rows], rows, harmonics=4)

    def test_dispatch_needs_no_online_ac_or_joint_menu(self):
        with (patch.object(Feeder33, 'ac', side_effect=AssertionError("online AC")),
              patch.object(Feeder33, 'ac_batch', side_effect=AssertionError("online AC")),
              patch('coordination_benchmark.reference_actions', side_effect=AssertionError("enumeration"))):
            result = self.model.dispatch(2, {8: 0, 30: 1}, q_limits={33: (-.05, 0.)})
            self.assertLessEqual(result.actions[2][1], 0.)
            self.assertTrue(np.isfinite(result.objective))
            self.assertEqual(result.diagnostics["messages"], 4)
            self.assertLessEqual(sum(abs(q) for _,q in result.actions), .05 + 1e-12)

    def test_fully_unobserved_and_invalid_telemetry(self):
        self.assertTrue(np.isfinite(self.model.dispatch(0, {}).objective))
        for bucket, evidence in ((4, {}), (1, {8: -1}), (1, {99: 0})):
            with self.assertRaises(ValueError):
                self.model.dispatch(bucket, evidence)


if __name__ == '__main__':
    unittest.main()
