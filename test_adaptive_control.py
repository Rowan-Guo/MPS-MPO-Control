import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from adaptive_control import qualify_dispatch
from controller import VoltageController
from coordination_benchmark import Scenario, TIMESHIFTS
from feeder33 import Feeder33, action_injections
from measured_profiles import DATA_PATH, construct_profiles, read_ausgrid
from powerflow import ACTIONS
from realistic_benchmark import DEADBAND
from temporal_control import TemporalModel


class TestAdaptiveControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feeder = Feeder33()
        _, load, pv = read_ausgrid(DATA_PATH)
        cls.powers = construct_profiles(load, pv, cls.feeder, day_lags=TIMESHIFTS)[:48*40]
        cls.times = np.arange(len(cls.powers))
        cls.model = VoltageController.fit(cls.feeder, Scenario('test'), cls.powers, cls.times,
                                          harmonics=4, calibration_samples=8)

    def test_qualification_replays_later_training_rows_with_exact_ac(self):
        archive = self.model.qualification
        self.assertGreater(len(archive.actions), 1)
        self.assertLessEqual(len(archive.actions), 48)
        np.testing.assert_array_equal(archive.source_rows, self.times[24*48:])
        for index in (0, len(archive.actions)-1):
            injections = [(b, *pq) for b, pq in zip(self.model.pcc_buses, archive.actions[index], strict=True)]
            rows = archive.source_rows[::53]
            v = abs(self.feeder.ac_batch(self.powers[rows], actions=injections))
            expected_loss = np.maximum(abs(v[:, [17, 32]]-1)-DEADBAND, 0.).mean(axis=1)
            np.testing.assert_allclose(archive.losses[index, ::53], expected_loss, atol=1e-9)
            np.testing.assert_allclose(archive.excursions[index, ::53], abs(v-1).max(axis=1), atol=1e-9)

    def test_reference_actions_have_zero_differential_response(self):
        for ref, response in self.model.responses.items():
            mapping = {b:(p, q) for b,p,q in action_injections(ACTIONS[ref])}
            for i, s in enumerate(self.model.stations):
                index = list(map(tuple, s.actions)).index(mapping.get(s.bus, (0., 0.)))
                np.testing.assert_array_equal(response.intercepts[i][index], 0.)
                np.testing.assert_array_equal(response.slopes[i][index], 0.)

    def test_archive_guard_and_physical_execution(self):
        model = self.model
        incumbent = model.dispatch(2, {}, method='legacy')
        archive = model.qualification
        # Construct paired records with a known best *safe* policy and a
        # superficially cheaper policy introducing a joint all-bus event.
        menus = {tuple(plan): i for i, plan in enumerate(archive.actions)}
        if incumbent.actions not in menus:
            self.fail("fixture requires a replayed incumbent")
        other = [i for i, plan in enumerate(archive.actions) if plan != incumbent.actions]
        base, safe, unsafe = menus[incumbent.actions], other[0], other[1]
        losses = np.full_like(archive.losses, .08)
        losses[base] = .04
        losses[safe] = .01
        losses[unsafe] = 0.
        excess = np.zeros_like(losses)
        excess[unsafe] = .051
        modified = replace(model, qualification=replace(archive, losses=losses, excursions=excess))
        with patch.object(Feeder33, 'ac_batch', side_effect=AssertionError("online AC")):
            result = qualify_dispatch(modified, incumbent, model.stations, 2, {}, 'local')
        self.assertEqual(result.actions, archive.actions[safe])
        self.assertEqual(result.diagnostics['messages'], 8)
        self.assertEqual(result.diagnostics['agreement_sweeps'], 1)
        for i, station in enumerate(model.stations):
            np.testing.assert_array_equal(station.actions[result.action_indices[i]], result.actions[i])
        self.assertGreater(result.diagnostics['wire_bytes'], incumbent.diagnostics['wire_bytes'])

    def test_temporal_fallback_and_conditional_probabilities(self):
        current = {8: 1}
        original = self.model.dispatch(2, current, method='adaptive')
        result = self.model.dispatch(2, current, method='temporal', history={1: {}})
        self.assertEqual(result, original)
        site = self.model.site_buses.index(30)
        a = self.model.predict_categories(2, current, history={1: {30: 0}}, half_hour=25)
        b = self.model.predict_categories(2, current, history={1: {30: 2}}, half_hour=25)
        np.testing.assert_allclose(a.sum(axis=1), 1.)
        np.testing.assert_array_equal(a[self.model.site_buses.index(8)], (0., 1., 0.))
        self.assertGreater(np.linalg.norm(a[site]-b[site]), .01)
        for args in (({3: {30: 1}}, 25), ({1: {99: 1}}, 25), ({1: {30: 2}}, 2)):
            with self.assertRaises(ValueError):
                self.model.predict_categories(2, current, history=args[0], half_hour=args[1])

    def test_lags_do_not_cross_missing_timestamps(self):
        indices = np.delete(self.times[:240], [7, 65])
        model = TemporalModel.fit(self.model.training_states[indices], self.model.training_buckets[indices],
                                  indices, self.model.mixtures)
        prev = model.predecessors[1]
        self.assertEqual(prev[np.flatnonzero(indices == 8)[0]], -1)
        for lag, lookup in model.predecessors.items():
            valid = lookup >= 0
            np.testing.assert_array_equal(indices[valid] - indices[lookup[valid]], lag)


if __name__ == '__main__':
    unittest.main()
