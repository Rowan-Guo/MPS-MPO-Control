import unittest

import numpy as np

from distributed import Station
from policy_agreement import agree_archive


class TestArchiveAgreement(unittest.TestCase):
    def test_private_availability_supersedes_terminal_preference(self):
        limits = ([True, True], [True, True], [True, False])
        zero, up = (0., 0.), (0., .025)
        stations = [Station(b, [zero, up], [[0.], [.01]], [0., 0.], available=available)
                    for b, available in zip((3, 6, 9), limits)]
        schedules = [(zero, zero, zero), (zero, zero, up), (zero, up, zero)]
        for transport in ('local', 'process'):
            selected, indices, counters = agree_archive(stations, schedules, [2., 0., 1.],
                                                        transport=transport)
            self.assertEqual((selected, indices), (2, (0, 1, 0)))
            self.assertEqual(counters["messages"], 4)
            self.assertEqual(counters["wire_bytes"], 4)

    def test_multiple_bytes_and_terminal_veto(self):
        actions = [(0., i*.001) for i in range(12)]
        stations = [Station(i, actions, [[0.]]*12, [0.]*12) for i in (1, 2)]
        schedules = [(a, a) for a in actions]
        scores = np.full(12, np.inf)
        scores[11] = -.01
        selected, indices, meta = agree_archive(stations, schedules, scores, transport="local")
        self.assertEqual((selected, indices), (11, (11, 11)))
        self.assertEqual(meta["wire_bytes"], 3)
        with self.assertRaisesRegex(ValueError, "available"):
            agree_archive(stations, schedules, np.full(12, np.inf), transport="local")
