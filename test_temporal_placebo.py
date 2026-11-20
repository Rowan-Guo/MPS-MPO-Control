import unittest
from types import SimpleNamespace

import numpy as np

from temporal_placebo import donor_rows, paired_replay
from topologies import LAYOUTS


class TestDonors(unittest.TestCase):
    def test_clock_match_joint_day_and_no_raw_overlap(self):
        train = np.arange(60*48)
        test = np.arange(80*48, 87*48)
        donor = donor_rows(train, test, 33, seed=3)
        np.testing.assert_array_equal(donor % 48, test % 48)
        np.testing.assert_array_equal(np.diff(donor.reshape(-1, 48), axis=1), 1)
        self.assertEqual(len(set(donor[::48]//48)), 7)
        self.assertLess(donor.max()+33*48, test.min())
        np.testing.assert_array_equal(donor_rows(train, test, 33, seed=3), donor)
        with self.assertRaises(ValueError):
            donor_rows(train, test, 83, seed=3)

    def test_identical_day_keeps_decisions_and_shares_context_cache(self):
        layout = LAYOUTS['ieee33']
        stations = [SimpleNamespace(actions=((0., 0.),), available=None, bus=bus) for bus in layout.controls]
        class Model:
            calls = 0
            def dispatch(self, clock_bucket, current, **kwargs):
                self.calls += 1
                return SimpleNamespace(actions=tuple((0., 0.) for _ in stations))
            def predict_categories(self, clock_bucket, current, **kwargs):
                if current:
                    raise AssertionError("pre-report metric leaked current readings")
                return np.full((len(layout.sites), 3), 1/3)
        model = Model()
        model.stations = stations
        times = np.arange(3*48)
        reports = (np.column_stack((times, times+1)) % 3).astype(int)
        for lag in (1, 48):
            model.calls = 0
            acts, prob, unique = paired_replay(model, layout, times, reports, reports.copy(), lag)
            np.testing.assert_array_equal(acts[0], acts[1])
            np.testing.assert_array_equal(prob[0], prob[1])
            self.assertEqual(model.calls, unique)
            self.assertLess(unique, len(times))


if __name__ == '__main__':
    unittest.main()
