import unittest

import numpy as np

from switch_replay import SWITCH_ORDER, evaluate_switches, switch_schedule
from topology_benchmark import Condition


class TestSwitchReplay(unittest.TestCase):
    def test_unordered_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            switch_schedule([48, 47])

    def test_both_policies_see_same_physical_grid_and_switches_follow_clock(self):
        heldout = np.arange(48*3)
        base = np.zeros((len(heldout), 3, 2))
        decisions = {(topo, "nominal"): base.copy() for topo in SWITCH_ORDER}
        gross = np.ones(48*40)
        solar = np.zeros_like(gross)
        def replay():
            return evaluate_switches(decisions, [Condition("nominal")], gross, solar, heldout, 3)
        result = replay()
        self.assertEqual([row["days"] for row in result["schedule"]], [1, 1, 1])
        case = result["conditions"]["nominal"]
        self.assertEqual(case["policies_full_window"]["status_updated"],
                         case["policies_full_window"]["status_frozen"])
        self.assertEqual(case["paired_after_first_switch"]["mean_monitor_loss"]["mean"], 0.)
        # Change a setpoint only in the *last* topology's model. It must take
        # effect in exactly that topology's 48 intervals and change the AC loss.
        decisions[SWITCH_ORDER[-1], "nominal"][:, 1, 1] = .025
        case = replay()["conditions"]["nominal"]
        self.assertEqual(case["changed_dispatches_after_first_switch"], 48)
        self.assertNotEqual(case["paired_after_first_switch"]["mean_monitor_loss"]["mean"], 0.)
