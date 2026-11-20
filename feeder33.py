"""Baran--Wu / MATPOWER case33bw radial network and nonlinear AC sweep.

This is a standard distribution test case with 33 buses, 12.66 kV and
10 MVA base, using the 32 normally-closed branches; five tie lines stay open.
Bus IDs exposed by the API are 1-based. Power inputs are in per unit.
Source: https://github.com/MATPOWER/matpower/blob/master/data/case33bw.m
"""
from __future__ import annotations
from dataclasses import dataclass
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
 [13,14,.5416,.7129],[14,15,.5916,.526],[15,16,.7463,.545],
 [16,17,1.289,1.721],[17,18,.732,.574],[2,19,.164,.1565],
 [19,20,1.5042,1.3554],[20,21,.4095,.4784],[21,22,.7089,.9373],
 [3,23,.4512,.3083],[23,24,.898,.7091],[24,25,.896,.7011],
 [6,26,.203,.1034],[26,27,.2842,.1447],[27,28,1.059,.9337],
 [28,29,.8042,.7006],[29,30,.5075,.2585],[30,31,.9744,.963],
 [31,32,.3105,.3619],[32,33,.341,.5302]], dtype=float)
BASE_MVA = 10.0
BASE_KV = 12.66

@dataclass(frozen=True)
class Feeder33:
    slack: float = 1.02
    base_load_scale: float = 0.65
    trunk_length_factor: float = 1.0

    def __post_init__(self):
        if not (0.95 <= self.slack <= 1.08 and 0.1 <= self.base_load_scale <= 2.0):
            raise ValueError("invalid feeder voltage/load scaling")
        if not np.isfinite(self.trunk_length_factor) or self.trunk_length_factor <= 0:
            raise ValueError("trunk length factor must be positive")

    @property
    def nominal_pq(self):
        return np.stack([P_KW,Q_KVAR], axis=1)* (self.base_load_scale / (BASE_MVA*1000))

    @cached_property
    def parent(self):
        parents=np.full(33,-1,dtype=int)
        for src,dst,*_ in CLOSED_BRANCHES:
            src,dst=int(src),int(dst)
            if parents[dst-1]>=0: raise ValueError("duplicate feeder parent")
            parents[dst-1]=src-1
        if parents[0]!=-1 or np.any(parents[1:]<0): raise ValueError("disconnected network")
        return parents

    @cached_property
    def impedance(self):
        z=np.zeros(33,complex)
        for src,dst,r,x in CLOSED_BRANCHES:
            # Extend both R and X on the path 1--18 only. Keep all lateral spans.
            length = self.trunk_length_factor if int(dst) <= 18 else 1.0
            z[int(dst)-1]=length*complex(r,x) / (BASE_KV**2 / BASE_MVA)
        return z

    def ac(self, power, *, actions=(), max_iterations=120, tolerance=1e-10):
        """Return 33 complex bus voltages using full nonlinear PQ backward/forward sweep.

        power: (33,2) nodal consumptions (P,Q), injections as negative loads.
        actions: (bus_id, active_injection_pu, reactive_injection_pu).
        """
        loads=np.asarray(power,dtype=float)
        if loads.shape != (33,2) or not np.all(np.isfinite(loads)):
            raise ValueError("power must have shape (33,2) and finite values")
        s=loads[:,0].astype(complex)+1j*loads[:,1]
        s[0]=0
        for bus,p,q in actions:
            if not 2<=bus<=33: raise ValueError("invalid controllable bus")
            s[bus-1]-=complex(p,q)
        z=self.impedance
        parent=self.parent
        v=np.full(33,complex(self.slack),dtype=complex)
        for _ in range(max_iterations):
            currents=np.conj(s/v)
            for k in range(32,0,-1):
                currents[parent[k]]+=currents[k]
            vnew=np.empty_like(v)
            vnew[0]=self.slack
            for k in range(1,33):
                vnew[k]=vnew[parent[k]]-z[k]*currents[k]
            if np.max(np.abs(vnew-v)) < tolerance:
                return vnew
            v=vnew
        raise RuntimeError("IEEE33 AC power flow failed to converge")

    def ac_batch(self, powers, *, actions=(), max_iterations=120, tolerance=1e-10):
        """Nonlinear AC solves for independent snapshots, vectorized over time.

        All snapshots receive the same optional list of PCC injections. For
        snapshot-specific dispatch the caller can subtract it from powers first.
        """
        loads = np.asarray(powers, dtype=float)
        if loads.ndim != 3 or loads.shape[1:] != (33, 2) or not np.isfinite(loads).all():
            raise ValueError("powers must have shape (samples,33,2) and finite values")
        # Keep all timestamps for a bus contiguous in memory during branch sweeps.
        s = np.ascontiguousarray((loads[:, :, 0] + 1j * loads[:, :, 1]).T)
        s[0] = 0
        for bus, p, q in actions:
            if not 2 <= bus <= 33 or not np.isfinite([p, q]).all():
                raise ValueError("invalid PCC injection")
            s[bus - 1] -= complex(p, q)
        v = np.full(s.shape, complex(self.slack))
        if not len(loads):
            return v.T
        z, parent = self.impedance, self.parent
        for _ in range(max_iterations):
            currents = np.conj(s / v)
            for k in range(32, 0, -1):
                currents[parent[k]] += currents[k]
            updated = np.empty_like(v)
            updated[0] = self.slack
            for k in range(1, 33):
                updated[k] = updated[parent[k]] - z[k] * currents[k]
            if np.max(np.abs(updated - v)) < tolerance:
                return updated.T
            v = updated
        raise RuntimeError("batched IEEE33 AC power flow failed to converge")


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
