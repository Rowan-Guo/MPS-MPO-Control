"""Baran--Wu / MATPOWER case33bw data and a radial nonlinear AC sweep.

This is a standard distribution test case with 33 buses, 12.66 kV and
10 MVA base, using the 32 normally-closed branches; five tie lines stay open.
Bus IDs exposed by the API are 1-based. Power inputs are in per unit.
Source: https://github.com/MATPOWER/matpower/blob/master/data/case33bw.m
"""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
import numpy as np

P_KW = np.array([0,100,90,120,60,60,200,200,60,60,45,60,60,120,60,60,60,90,90,90,90,90,90,420,420,60,60,60,120,200,150,210,60], dtype=float)
Q_KVAR = np.array([0,60,40,80,30,20,100,100,20,20,30,35,35,80,10,20,20,40,40,40,40,40,50,200,200,25,25,20,70,600,70,100,40], dtype=float)
# normally closed lines in MATPOWER case33bw; (from, to, r_ohm, x_ohm)
CLOSED_BRANCHES = np.array([
 [1,2,.0922,.0470],[2,3,.493,.2511],[3,4,.366,.1864],
 [4,5,.3811,.1941],[5,6,.819,.707],[6,7,.1872,.6188],
 [7,8,.7114,.2351],[8,9,1.03,.74],[9,10,1.044,.74],
 [10,11,.1966,.065],[11,12,.3744,.1238],[12,13,1.468,1.155],
 [13,14,.5416,.7129],[14,15,.5910,.526],[15,16,.7463,.545],
 [16,17,1.289,1.721],[17,18,.732,.574],[2,19,.164,.1565],
 [19,20,1.5042,1.3554],[20,21,.4095,.4784],[21,22,.7089,.9373],
 [3,23,.4512,.3083],[23,24,.898,.7091],[24,25,.896,.7011],
 [6,26,.203,.1034],[26,27,.2842,.1447],[27,28,1.059,.9337],
 [28,29,.8042,.7006],[29,30,.5075,.2585],[30,31,.9744,.963],
 [31,32,.3105,.3619],[32,33,.341,.5302]], dtype=float)
BASE_MVA = 10.0
BASE_KV = 12.66

@dataclass(frozen=True, eq=False)
class FeederCase:
    name: str
    demands_kw_kvar: np.ndarray
    branches_ohm: np.ndarray
    trunk: tuple[int, ...]
    source_file: str
    closed_tie: tuple[int, int] | None = None
    opened_section: tuple[int, int] | None = None


CASE33 = FeederCase("ieee33", np.column_stack((P_KW, Q_KVAR)), CLOSED_BRANCHES,
                    tuple(range(1, 19)), "case33bw.m")
MATPOWER_DATA = ("https://github.com/MATPOWER/matpower/blob/"
                 "b6c85b0323d20fb801cab3377c803a285d730847/data/")


@dataclass(frozen=True)
class RadialFeeder:
    slack: float = 1.02
    base_load_scale: float = 0.65
    trunk_length_factor: float = 1.0
    case: FeederCase = field(default=CASE33, kw_only=True, repr=False)

    def __post_init__(self):
        if not (0.95 <= self.slack <= 1.08 and 0.1 <= self.base_load_scale <= 2.0):
            raise ValueError("invalid feeder voltage/load scaling")
        if not np.isfinite(self.trunk_length_factor) or self.trunk_length_factor <= 0:
            raise ValueError("trunk length factor must be positive")

    @property
    def n_buses(self):
        return len(self.case.demands_kw_kvar)

    @property
    def nominal_pq(self):
        return self.case.demands_kw_kvar * (self.base_load_scale / (BASE_MVA * 1000))

    @cached_property
    def _tree(self):
        """Orient the undirected feeder from slack bus 1, allowing arbitrary bus order."""
        n = self.n_buses
        edges = self.case.branches_ohm
        if edges.shape != (n - 1, 4) or not np.isfinite(edges).all():
            raise ValueError("a radial feeder needs N-1 finite branches")
        trunk_edges = {tuple(sorted(edge)) for edge in zip(self.case.trunk, self.case.trunk[1:])}
        neighbors = [[] for _ in range(n)]
        for a, b, r, x in edges:
            if not (a == int(a) and b == int(b) and 1 <= a <= n and 1 <= b <= n
                    and a != b and r >= 0 and x >= 0 and r + x > 0):
                raise ValueError("invalid feeder branch")
            left, right = int(a) - 1, int(b) - 1
            z = complex(r, x) / (BASE_KV**2 / BASE_MVA)
            if tuple(sorted((int(a), int(b)))) in trunk_edges:
                z *= self.trunk_length_factor
            neighbors[left].append((right, z))
            neighbors[right].append((left, z))

        parent = np.full(n, -1, dtype=int)
        impedance = np.zeros(n, dtype=complex)
        order = [0]
        for bus in order:
            for other, z in neighbors[bus]:
                if other != 0 and parent[other] < 0:
                    parent[other] = bus
                    impedance[other] = z
                    order.append(other)
        if len(order) != n:
            raise ValueError("closed branches must form a connected tree from bus 1")
        # Keep the forward order as ordinary Python integers inside the hot AC loop.
        return parent, impedance, tuple(order[1:])

    @property
    def parent(self):
        """Zero-based parent indices in bus order, with -1 at the slack."""
        return self._tree[0]

    @property
    def impedance(self):
        """Per-unit impedance to the oriented parent at each bus."""
        return self._tree[1]

    @cached_property
    def _paths(self):
        depth = np.zeros(self.n_buses, dtype=int)
        path_z = np.zeros(self.n_buses, dtype=complex)
        for bus in self._tree[2]:
            parent = self.parent[bus]
            depth[bus] = depth[parent] + 1
            path_z[bus] = path_z[parent] + self.impedance[bus]
        return depth, path_z

    def path_impedance(self, bus):
        """Sum of complex series impedances on the unique slack-to-bus path (pu)."""
        if not isinstance(bus, (int, np.integer)) or not 1 <= bus <= self.n_buses:
            raise ValueError("invalid bus ID")
        return self._paths[1][bus - 1]

    @property
    def topology_metadata(self):
        depths, path_z = self._paths
        longest = int(np.abs(path_z).argmax())
        children = np.bincount(self.parent[1:], minlength=self.n_buses)
        return {"topology": self.case.name, "buses": self.n_buses,
                "closed_branch_count": len(self.case.branches_ohm),
                "closed_tie": self.case.closed_tie, "opened_section": self.case.opened_section,
                "max_depth": int(depths.max()), "max_path_impedance_pu": float(abs(path_z[longest])),
                "max_path_impedance_bus": longest + 1, "leaf_buses": (np.flatnonzero(children == 0) + 1).tolist(),
                "base_mva": BASE_MVA, "base_kv": BASE_KV, "source_url": MATPOWER_DATA + self.case.source_file}

    def _apply_actions(self, s, actions):
        for bus, p, q in actions:
            if (not isinstance(bus, (int, np.integer)) or not 2 <= bus <= self.n_buses
                    or not np.isfinite([p, q]).all()):
                raise ValueError("invalid PCC injection")
            s[bus - 1] -= complex(p, q)

    def _sweep(self, s, max_iterations, tolerance):
        """Sweep a bus vector or bus-major batch until every voltage converges."""
        parent, impedance, order = self._tree
        v = np.full(s.shape, complex(self.slack))
        if not v.size:
            return v
        for _ in range(max_iterations):
            currents = np.conj(s / v)
            for bus in reversed(order):
                currents[parent[bus]] += currents[bus]
            updated = np.empty_like(v)
            updated[0] = self.slack
            for bus in order:
                updated[bus] = updated[parent[bus]] - impedance[bus] * currents[bus]
            if np.max(np.abs(updated - v)) < tolerance:
                return updated
            v = updated
        raise RuntimeError(f"{self.case.name} AC power flow failed to converge")

    def ac(self, power, *, actions=(), max_iterations=120, tolerance=1e-10):
        """Complex voltages for nodal consumption (P,Q) and optional PCC injections, in pu."""
        loads = np.asarray(power, dtype=float)
        if loads.shape != (self.n_buses, 2) or not np.isfinite(loads).all():
            raise ValueError(f"power must have shape ({self.n_buses},2) and finite values")
        s = loads[:, 0] + 1j * loads[:, 1]
        s[0] = 0
        self._apply_actions(s, actions)
        return self._sweep(s, max_iterations, tolerance)

    def ac_batch(self, powers, *, actions=(), max_iterations=120, tolerance=1e-10):
        """Batch independent PQ snapshots, with a common optional dispatch at every time."""
        loads = np.asarray(powers, dtype=float)
        if loads.ndim != 3 or loads.shape[1:] != (self.n_buses, 2) or not np.isfinite(loads).all():
            raise ValueError(f"powers must have shape (samples,{self.n_buses},2) and finite values")
        s = np.ascontiguousarray((loads[:, :, 0] + 1j * loads[:, :, 1]).T)
        s[0] = 0
        self._apply_actions(s, actions)
        return self._sweep(s, max_iterations, tolerance).T


class Feeder33(RadialFeeder):
    """Historical public entry point for the normally closed 33-bus test case."""


def action_injections(action, control_buses=(18,33)):
    """Split total (P,Q) dispatch equally between two feeder-end inverters."""
    p,q=action
    return tuple((bus,p/len(control_buses),q/len(control_buses)) for bus in control_buses)


def monitored_loss(voltages, monitored=(18,33), *, deadband=0.025):
    v=np.abs(np.asarray(voltages))
    return float(np.mean(np.maximum(0., np.abs(v[np.asarray(monitored)-1]-1.)-deadband)))


def unsafe_voltage(voltages, vmin=0.95, vmax=1.05):
    v=np.abs(voltages)
    return bool(v.min() < vmin or v.max() > vmax)
