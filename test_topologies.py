"""Cross-check feeder orientation, switch status, power balance and the 69-bus base."""
import unittest
from dataclasses import replace

import numpy as np

from feeder33 import BASE_KV, BASE_MVA, CASE33, RadialFeeder
from topologies import SWITCHES, TOPOLOGIES, make_feeder


def nodal_reference(feeder, loads):
    """Independent PQ solution by dense nodal admittance, at nominal branch length."""
    n = len(loads)
    y = np.zeros((n, n), dtype=complex)
    for left, right, r, x in feeder.case.branches_ohm:
        a, b = int(left) - 1, int(right) - 1
        admittance = (BASE_KV**2 / BASE_MVA) / complex(r, x)
        y[a, a] += admittance
        y[b, b] += admittance
        y[a, b] -= admittance
        y[b, a] -= admittance
    z_reduced = np.linalg.solve(y[1:, 1:], np.eye(n - 1))
    s = loads[1:, 0] + 1j * loads[1:, 1]
    v = np.full(n - 1, complex(feeder.slack))
    for _ in range(120):
        updated = feeder.slack - z_reduced @ np.conj(s / v)
        if np.max(np.abs(updated - v)) < 1e-12:
            return np.r_[feeder.slack, updated]
        v = updated
    raise RuntimeError("nodal reference failed to converge")


class TestRadialCases(unittest.TestCase):
    def test_two_switch_configurations(self):
        original = make_feeder("ieee33")
        edges = {tuple(sorted(row[:2])) for row in original.case.branches_ohm}
        for topology, (tie, opened) in SWITCHES.items():
            with self.subTest(topology=topology):
                f = make_feeder(topology)
                closed = {tuple(sorted(row[:2])) for row in f.case.branches_ohm}
                self.assertEqual(closed - edges, {tie[:2]})
                self.assertEqual(edges - closed, {opened})
                self.assertEqual(f.topology_metadata["closed_branch_count"], f.n_buses - 1)
                np.testing.assert_array_equal(f.nominal_pq, original.nominal_pq)
                self.assertTrue(any(f.parent[bus] > bus for bus in range(1, f.n_buses)))

    def test_nodal_reference_with_injection_and_batch(self):
        for topology in TOPOLOGIES:
            with self.subTest(topology=topology):
                f = make_feeder(topology)
                bus = 65 if topology == "case69" else 18
                injection = (bus, .022, .017)
                powers = np.array([f.nominal_pq * k for k in (.8, 1.2)])
                batch = f.ac_batch(powers, actions=[injection])
                for loads, voltages in zip(powers, batch):
                    scalar = f.ac(loads, actions=[injection])
                    physical = loads.copy()
                    physical[bus - 1] -= injection[1:]
                    np.testing.assert_allclose(scalar, nodal_reference(f, physical), rtol=0, atol=2e-10)
                    np.testing.assert_allclose(scalar, voltages, rtol=0, atol=2e-10)
                    # Independently reconstruct all local net powers from branch currents.
                    i_branch = (scalar[f.parent[1:]] - scalar[1:]) / f.impedance[1:]
                    i_local = np.r_[0j, i_branch]
                    np.subtract.at(i_local, f.parent[1:], i_branch)
                    np.testing.assert_allclose(scalar[1:] * np.conj(i_local[1:]),
                                               physical[1:, 0] + 1j * physical[1:, 1],
                                               rtol=0, atol=2e-9)

    def test_case69_source_units_and_full_load(self):
        f = make_feeder("case69", slack=1., base_load_scale=1.)
        self.assertEqual(f.n_buses, 69)
        np.testing.assert_allclose(f.nominal_pq.sum(axis=0) * 10000, [3802.1, 2694.7])
        v = f.ac(f.nominal_pq)
        current = (v[f.parent[1:]] - v[1:]) / f.impedance[1:]
        line_loss_kw = np.sum(np.abs(current)**2 * f.impedance[1:].real) * BASE_MVA * 1000
        self.assertEqual(np.argmin(abs(v)) + 1, 65)
        self.assertAlmostEqual(abs(v[64]), .909188, places=6)
        self.assertAlmostEqual(line_loss_kw, 224.992, places=3)
        long = make_feeder("case69", trunk_length_factor=1.5)
        self.assertAlmostEqual(long.path_impedance(65), 1.5 * f.path_impedance(65))
        # Bus 10 leaves the stretched trunk at 9. Its 9–10 conductor remains unchanged.
        self.assertAlmostEqual(long.impedance[9], f.impedance[9])

    def test_invalid_connectivity_and_failed_convergence(self):
        branches = CASE33.branches_ohm.copy()
        branches[0] = branches[-1]  # N-1 branches, but bus 1 is now isolated
        broken = RadialFeeder(case=replace(CASE33, branches_ohm=branches))
        with self.assertRaisesRegex(ValueError, "connected tree"):
            broken.ac(broken.nominal_pq)
        f = make_feeder("ieee33_tie_18_33")
        with self.assertRaises(RuntimeError):
            f.ac_batch(f.nominal_pq[None], max_iterations=1)
        self.assertEqual(f.ac_batch(np.empty((0, 33, 2))).shape, (0, 33))
        with self.assertRaises(ValueError):
            make_feeder("missing")
