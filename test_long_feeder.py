import unittest

import numpy as np

from feeder33 import Feeder33, action_injections
from long_feeder import reactive_sensitivities, weekly_interval


class TestLongFeeder(unittest.TestCase):
    def test_lengths_change_only_main_trunk(self):
        base = Feeder33()
        long = Feeder33(trunk_length_factor=2.)
        np.testing.assert_allclose(long.nominal_pq, base.nominal_pq)
        np.testing.assert_allclose(long.impedance[1:18], 2 * base.impedance[1:18])
        np.testing.assert_allclose(long.impedance[18:], base.impedance[18:])
        np.testing.assert_equal(long.parent, base.parent)
        with self.assertRaises(ValueError):
            Feeder33(trunk_length_factor=0.)

    def test_batch_and_reference_sweeps_agree(self):
        feeder = Feeder33(trunk_length_factor=1.7)
        powers = np.stack([feeder.nominal_pq * k for k in (.8, 1., 1.25)])
        actions = action_injections((.05, -.05))
        expected = [feeder.ac(power, actions=actions) for power in powers]
        np.testing.assert_allclose(feeder.ac_batch(powers, actions=actions), expected, atol=2e-10, rtol=0)
        self.assertEqual(feeder.ac_batch(powers[:0]).shape, (0, 33))
        np.testing.assert_allclose(weekly_interval(np.ones(40), np.arange(40)*24, 1), [1.,1.])

    def test_distal_reactive_leverage(self):
        normal = reactive_sensitivities(Feeder33())
        long = reactive_sensitivities(Feeder33(trunk_length_factor=2.))
        self.assertGreater(long["18"][2], normal["18"][2])
        self.assertGreater(long["18"][2], long["33"][2])


if __name__ == "__main__":
    unittest.main()
