"""Published radial feeder cases and two feasible switch configurations.

Both switch studies retain the original 33-bus injections, conductor parameters,
and device locations. Switch status is an experiment input fixed for each run.
Data provenance and physical units are recorded in data/README.md.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from feeder33 import CASE33, FeederCase, RadialFeeder
from measured_profiles import SITES


# (normally open tie incl. impedance in ohms, normally closed section to open)
SWITCHES = {
    "ieee33_tie_8_21": ((8, 21, 2., 2.), (7, 8)),
    "ieee33_tie_18_33": ((18, 33, .5, .5), (13, 14)),
}
TOPOLOGIES = ("ieee33", *SWITCHES, "case69")


@dataclass(frozen=True)
class Layout:
    sites: tuple
    controls: tuple
    monitors: tuple
    meters: tuple


IEEE33_LAYOUT = Layout(SITES, (14, 18, 33), (18, 33), (8, 30))
LAYOUTS = {
    "ieee33": IEEE33_LAYOUT,
    "ieee33_tie_8_21": IEEE33_LAYOUT,
    "ieee33_tie_18_33": IEEE33_LAYOUT,
    "case69": Layout((8, 11, 12, 16, 18, 21, 24, 49, 50, 61, 64, 65),
                     (21, 27, 65), (27, 65), (12, 64)),
}


@lru_cache(maxsize=len(TOPOLOGIES))
def _case(topology):
    if topology == "ieee33":
        return CASE33
    if topology in SWITCHES:
        tie, section = SWITCHES[topology]
        closed = CASE33.branches_ohm
        opened = (closed[:, :2] == section).all(axis=1)
        if opened.sum() != 1:
            raise ValueError(f"section {section} missing from nominal feeder")
        return replace(CASE33, name=topology, branches_ohm=np.vstack((closed[~opened], tie)),
                       closed_tie=tie[:2], opened_section=section)
    if topology == "case69":
        table = np.loadtxt(Path(__file__).parent / "data" / "case69.csv", delimiter=",", skiprows=1)
        if not np.array_equal(table[:, 0], np.arange(1, 70)):
            raise ValueError("case69 source rows must be in bus ID order")
        # CSV has one receiving branch per non-slack bus, in physical units.
        branches = table[1:, [1, 0, 2, 3]]
        trunk = (*range(1, 10), *range(53, 66))  # physical path to the weak end at bus 65
        return FeederCase(topology, table[:, 4:6], branches, trunk, "case69.m")
    raise ValueError(f"unknown topology {topology!r}, expected one of {TOPOLOGIES}")


def make_feeder(topology, *, slack=1.02, base_load_scale=.65, trunk_length_factor=1.):
    """Build a radial case. Scaling extends both R and X on fixed physical trunk spans."""
    return RadialFeeder(slack, base_load_scale, trunk_length_factor, case=_case(topology))
