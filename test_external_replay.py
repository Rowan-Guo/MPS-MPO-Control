"""Confirm the meter units, time alignment and training boundary before replay."""
import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import external_replay as ext


class ExternalReplayTests(unittest.TestCase):
    def test_meter_year_and_start_of_interval_convention(self):
        times, gross, pv = ext.read_daily_meter()
        self.assertEqual(len(times), 365*48)
        self.assertEqual(str(times[0]), "2012-07-01T00:00")
        self.assertEqual(str(times[-1]), "2013-06-30T23:30")
        np.testing.assert_array_equal(np.diff(times), np.timedelta64(30, "m"))
        # Source row contains 0.241 and 0.197 kWh for intervals ending at 0:30 and 1:00.
        np.testing.assert_allclose(gross[:2], [.482, .394])
        self.assertTrue(np.all(pv >= 0))

    def test_replay_fits_only_source_year_and_reuses_its_scales(self):
        train_times, gross, pv = ext.read_ausgrid()
        scale = ext.profile_scales(gross, pv)
        lags = tuple(range(0, 36, 3))
        layout = ext.LAYOUTS["ieee33"]
        expected = ext.construct_profiles(gross, pv, ext.make_feeder("ieee33"), sites=layout.sites,
                                          day_lags=lags, solar_scale=1.5)[:160*48]
        with patch.object(ext, "construct_profiles", wraps=ext.construct_profiles) as construct, \
             patch.object(ext.VoltageController, "fit") as fit, \
             patch.object(ext, "evaluate_case", side_effect=StopIteration):
            with self.assertRaises(StopIteration):
                ext.replay()
        self.assertEqual(fit.call_count, 1)
        np.testing.assert_array_equal(fit.call_args.args[2], expected)
        np.testing.assert_array_equal(fit.call_args.args[3], np.arange(160*48))
        self.assertEqual(len(construct.call_args_list), 2)
        for call in construct.call_args_list:
            self.assertEqual(call.kwargs["reference_scales"], scale)
        self.assertEqual(len(construct.call_args_list[1].args[0]), 365*48)
        self.assertLess(train_times[-1], ext.read_daily_meter()[0][0])

    def test_temporal_protocol_uses_two_nominal_grids_and_all_consecutive_dates(self):
        calls = []
        def evaluate(feeder, model, layout, indices, snapshots, *, blackout, alternating, seed):
            self.assertFalse(alternating)
            calls.append((feeder.case.name, feeder.slack, blackout))
            np.testing.assert_array_equal(indices, np.arange(332*48))
            self.assertEqual(len(snapshots), len(indices))
            return {}, {"half_hour": {"all_bus_unsafe_fraction": np.zeros(len(indices))}}

        stub = SimpleNamespace(evaluate_temporal=evaluate)
        with patch.dict(sys.modules, {"temporal_benchmark": stub}), patch.object(ext.VoltageController, "fit"):
            report = ext.replay(temporal="blackout")
        self.assertEqual(calls, [("ieee33", 1.02, True), ("case69", 1.02, True)])
        self.assertEqual(report["cases"], 2)
        self.assertEqual(report["macro_memoryless_minus_lag"]["half_hour"]["all_bus_unsafe_fraction"]["mean"], 0.)
