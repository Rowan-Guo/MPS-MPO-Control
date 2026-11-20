import unittest
from types import SimpleNamespace

import numpy as np

from distributed import Station
from temporal_benchmark import available_reports, evaluate_temporal, lag_reports, replay_policy
from topologies import IEEE33_LAYOUT


class RecordingModel:
    def __init__(self):
        layout = IEEE33_LAYOUT
        self.stations = [Station(bus, [(0., 0.), (0., .025)], [[0.], [.001]], [0., 0.])
                         for bus in layout.controls]
        self.thresholds = np.tile([.3, .7], (len(layout.sites), 1))
        self.dispatched, self.predicted = [], []

    def dispatch(self, bucket, observed, *, method, history=None, half_hour=None):
        self.dispatched.append((bucket, observed, history, half_hour))
        # Make actions visibly depend on history rather than on hidden truth.
        return SimpleNamespace(actions=[(0., .025 if history else 0.), (0., 0.), (0., 0.)])

    def predict_categories(self, bucket, observed, *, history=None, half_hour=None):
        self.predicted.append((bucket, observed, history, half_hour))
        row = [.5, .25, .25] if observed else [1/3] * 3
        return np.tile(row, (len(IEEE33_LAYOUT.sites), 1))


class TestTemporalReplay(unittest.TestCase):
    def test_buffer_never_exposes_hidden_or_preperiod_categories(self):
        clock = np.arange(96, 96+96)
        levels = np.repeat(np.arange(len(clock))[:, None] % 3, 2, axis=1)
        reports, missing = available_reports(clock, levels, True)
        self.assertEqual(missing.sum(), 48)
        changed = levels.copy()
        changed[missing] = (changed[missing] + 1) % 3
        np.testing.assert_array_equal(available_reports(clock, changed, True)[0], reports)
        self.assertTrue((lag_reports(reports, 48)[:48] == -1).all())
        self.assertTrue((lag_reports(reports, 1)[np.flatnonzero(missing) + 1] == -1).all())
        for lag in (0, 1, 48):
            model = RecordingModel()
            chosen, _, _, previous, _ = replay_policy(model, IEEE33_LAYOUT, clock, reports, lag)
            np.testing.assert_array_equal(chosen[:, 0, 1] > 0, np.any(previous >= 0, axis=1))
            self.assertTrue(all(set(current) <= set(IEEE33_LAYOUT.meters) for _, current, _, _ in model.predicted))
            for _, observed, history, now in model.dispatched:
                if now is not None:
                    pos = int(now-clock[0])
                    self.assertEqual(bool(observed), not missing[pos])
                    self.assertGreaterEqual(pos, lag)
                    self.assertFalse(missing[pos-lag])
                    self.assertEqual(tuple(history[lag].values()), tuple(levels[pos-lag]))

    def test_common_ac_outcomes_and_log_score_cover_all_intervals(self):
        layout, model = IEEE33_LAYOUT, RecordingModel()
        loads = np.zeros((96, 33, 2))
        loads[:, layout.meters[0]-1, 0] = np.linspace(0., 1., len(loads))
        snapshots = loads.copy()
        # A common non-controlled weak bus yields an all-bus event for every method.
        def ac(powers):
            v = np.ones(powers.shape[:2])
            v[:, 3] = .94
            return v
        result, deltas = evaluate_temporal(SimpleNamespace(ac_batch=ac), model, layout,
                                           np.arange(96, 192), snapshots, blackout=True)
        self.assertEqual(result["no_current_report_intervals"], 48)
        for metric in result["methods"].values():
            self.assertEqual(metric["all_bus_unsafe_fraction"], 1.)
            self.assertAlmostEqual(metric["pre_report_meter_nll"], np.log(3.))
            self.assertAlmostEqual(metric["post_report_hidden_site_nll"], (np.log(2.) + np.log(3.)) / 2)
        for policy in deltas.values():
            for delta in policy.values():
                np.testing.assert_array_equal(delta, np.zeros(96))
        np.testing.assert_array_equal(loads, snapshots)
        with self.assertRaises(ValueError):
            available_reports([0, 2], np.zeros((2, 2)), False)

    def test_alternation_uses_only_the_other_channels_previous_report(self):
        clock = np.arange(96)
        values = np.column_stack((clock % 3, (clock + 1) % 3))
        reports, missing = available_reports(clock, values, blackout=False, alternating=True)
        self.assertFalse(missing.any())
        np.testing.assert_array_equal(np.sum(reports >= 0, axis=1), np.ones(len(clock)))
        np.testing.assert_array_equal(reports[::2, 0], values[::2, 0])
        np.testing.assert_array_equal(reports[1::2, 1], values[1::2, 1])
        for lag in (1, 48):
            previous = lag_reports(reports, lag)
            np.testing.assert_array_equal(previous[lag:] >= 0,
                                          (reports[lag:] < 0) if lag == 1 else (reports[lag:] >= 0))


if __name__ == "__main__":
    unittest.main()
