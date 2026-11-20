"""Compare the same trained control rule with current versus stale switch status."""
import numpy as np

from measured_profiles import construct_profiles
from topologies import make_feeder


SWITCH_ORDER = ("ieee33", "ieee33_tie_8_21", "ieee33_tie_18_33")


def switch_schedule(heldout):
    """One preassigned configuration per block of whole days, in chronological order."""
    heldout = np.asarray(heldout)
    if heldout.ndim != 1 or not len(heldout) or np.any(np.diff(heldout) <= 0):
        raise ValueError("held-out half-hours must be strictly increasing")
    days = heldout // 48
    return [np.flatnonzero(np.isin(days, block))
            for block in np.array_split(np.unique(days), len(SWITCH_ORDER))]


def evaluate_switches(decisions, conditions, gross, solar, heldout, seed):
    from topology_benchmark import IEEE33_LAYOUT, paired_effect, voltage_scores

    layout = IEEE33_LAYOUT
    rows_by_grid = switch_schedule(heldout)
    lags = tuple(range(0, 3*len(layout.sites), 3))
    conditions_out = {}
    for condition in conditions:
        nominal = make_feeder(SWITCH_ORDER[0], slack=condition.slack,
                              base_load_scale=condition.load)
        snapshots = construct_profiles(gross, solar, nominal, sites=layout.sites,
                                       day_lags=lags, solar_scale=condition.pv)[heldout]
        voltages = {name: np.empty((len(heldout), nominal.n_buses), complex)
                    for name in ("status_updated", "status_frozen")}
        distinct = np.zeros(len(heldout), bool)
        for topo, rows in zip(SWITCH_ORDER, rows_by_grid, strict=True):
            if not len(rows):
                continue
            physical = make_feeder(topo, slack=condition.slack, base_load_scale=condition.load)
            updated = decisions[topo, condition.name][rows]
            frozen = decisions[SWITCH_ORDER[0], condition.name][rows]
            distinct[rows] = np.any(updated != frozen, axis=(1, 2))
            for name, actions in (("status_updated", updated), ("status_frozen", frozen)):
                if name == "status_frozen" and not distinct[rows].any():
                    voltages[name][rows] = voltages["status_updated"][rows]
                    continue
                loads = snapshots[rows].copy()
                loads[:, np.asarray(layout.controls)-1] -= actions
                voltages[name][rows] = physical.ac_batch(loads)

        summaries, losses, unsafe = {}, {}, {}
        for name, trace in voltages.items():
            summaries[name], losses[name], unsafe[name] = voltage_scores(trace, layout.monitors)
        after = np.concatenate(rows_by_grid[1:])
        paired = {}
        if len(after):
            for name, values in (("mean_monitor_loss", losses), ("all_bus_unsafe_fraction", unsafe)):
                delta = values["status_frozen"][after] - values["status_updated"][after]
                paired[name] = paired_effect(delta, np.asarray(heldout)[after], seed)
        conditions_out[condition.name] = dict(
            policies_full_window=summaries, paired_after_first_switch=paired,
            changed_dispatches_after_first_switch=int(distinct[after].sum()),
            intervals_after_first_switch=len(after))
    return dict(
        protocol="Three consecutive, prespecified blocks of whole held-out days. Both policies use "
                 "the adaptive solver and identical inputs except the switch status. Updated selects "
                 "the pre-calibrated model for the actual grid; frozen always selects the nominal "
                 "grid model. Both sets of commands are replayed on the actual switched feeder. "
                 "Steady-state half-hour AC; switch transients and status-detection delay excluded.",
        schedule=[dict(topology=topo, days=int(len(np.unique(np.asarray(heldout)[rows]//48))),
                       intervals=len(rows)) for topo, rows in zip(SWITCH_ORDER, rows_by_grid, strict=True)],
        conditions=conditions_out)
