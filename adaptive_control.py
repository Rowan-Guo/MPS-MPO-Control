"""Reference-anchored, operating-point-dependent local transitions for min-sum."""
from dataclasses import dataclass, replace

import numpy as np

from coordination_benchmark import BUDGET
from distributed import solve_dispatch
from fast_fourier import FactorizedFourier, affine_bounds
from feeder33 import action_injections
from powerflow import ACTIONS
from realistic_benchmark import DEADBAND


@dataclass
class ResponseModel:
    """For each station, predict change in |V| from its reference action.

    Each differential response is intercept + slope * (reference_voltage - center).
    The voltage predictor is local to the receiving bus, even when the change
    originates on another feeder branch.
    """
    center: np.ndarray
    intercepts: tuple
    slopes: tuple
    operator: FactorizedFourier


def calibrate_responses(feeder, powers, stations, split_buses, monitors, references,
                        constants, contributions, harmonics, tick, samples=32):
    """Train single-station responses around each available equal-split preset.

    Sample quantiles of each receiving monitor's training voltage, preserving
    both low- and high-voltage operating conditions. No joint action table.
    """
    if samples < 2:
        raise ValueError("response calibration requires at least two samples")
    monitor_ids = np.asarray(monitors) - 1
    models = {}
    for ref, reference in references.items():
        injections = action_injections(ACTIONS[ref], split_buses)
        targets = {b: (p, q) for b, p, q in injections}
        if any(targets.get(s.bus, (0., 0.)) not in list(map(tuple, s.actions))
               for s in stations):
            continue
        quantiles = np.linspace(0, len(reference)-1, min(samples, len(reference)), dtype=int)
        rows = np.unique(np.argsort(reference[:, monitor_ids], axis=0, kind='stable')[quantiles])
        base = powers[rows].copy()
        for bus, p, q in injections:
            base[:, bus-1] -= (p, q)
        xr = reference[rows]
        center = xr.mean(axis=0)
        x = xr - center
        variance = np.sum(x*x, axis=0)
        intercepts, slopes = [], []
        # Bound every shift seen by the Fourier prior, including per-station rounding.
        lower_shift = np.zeros(len(monitor_ids))
        upper_shift = lower_shift.copy()
        xmin = reference.min(axis=0) - center
        xmax = reference.max(axis=0) - center
        for s in stations:
            actions = np.asarray(s.actions)
            deltas = actions - targets.get(s.bus, (0., 0.))
            probe = np.broadcast_to(base, (len(actions), *base.shape)).copy()
            probe[:, :, s.bus-1] -= deltas[:, None, :]
            magnitudes = np.abs(feeder.ac_batch(probe.reshape(-1, *base.shape[1:])))
            delta = magnitudes.reshape(len(actions), *xr.shape) - xr
            mean = delta.mean(axis=1)
            slope = np.divide(np.einsum('arb,rb->ab', delta, x), variance,
                              out=np.zeros_like(mean), where=variance > 1e-12)
            unchanged = np.all(deltas == 0, axis=1)
            mean[unchanged] = 0.
            slope[unchanged] = 0.
            intercepts.append(mean)
            slopes.append(slope)
            at_low = (mean + slope*xmin)[:, monitor_ids]
            at_high = (mean + slope*xmax)[:, monitor_ids]
            lower_shift += np.minimum(at_low, at_high).min(axis=0) - tick
            upper_shift += np.maximum(at_low, at_high).max(axis=0) + tick
        c = constants[ref:ref+1]
        f = contributions[ref:ref+1]
        low, high = affine_bounds(c, f)
        bound = max(np.abs(low+lower_shift).max(), np.abs(high+upper_shift).max(), DEADBAND)
        operator = FactorizedFourier(c, f, DEADBAND, harmonics, domain=1.05*bound)
        models[ref] = ResponseModel(center, tuple(intercepts), tuple(slopes), operator)
    return models


def select_training_context(states, buckets, bucket, evidence):
    rows = np.flatnonzero(buckets == bucket)
    distance = np.zeros(len(rows), int)
    for site, level in evidence.items():
        distance += states[rows, site] != level
    nearest = distance.min()
    return rows[distance == nearest], int(np.sum(distance == 0))


def adaptive_dispatch(stations, action_rows, model, reference_voltages, no_action_voltages,
                      rows, matches, mixture, evidence, monitors, tick, transport,
                      *, snapshot_proposal=False):
    """Pass two local voltage-response coefficients where needed, plus fleet ledgers.

    Terminal loss uses the conditional Fourier MPO prior and an anchored,
    sample-dependent empirical loss. The paired safety screen compares joint
    all-bus voltage events against the feasible reference on the same rows.
    """
    v = reference_voltages[rows]
    center = v.mean(axis=0)
    spread = np.max(np.abs(v-center), axis=0)
    normalized = np.divide(v-center, spread, out=np.zeros_like(v), where=spread > 0)
    offsets, variations = [], []
    low, high = v.min(axis=0), v.max(axis=0)
    for station_idx, idx in enumerate(action_rows):
        slope = model.slopes[station_idx][idx]
        offset = model.intercepts[station_idx][idx] + slope*(center-model.center)
        midpoint = np.rint(offset/tick)*tick
        drift = np.rint(slope*spread/tick)*tick
        offsets.append(midpoint)
        variations.append(drift)
        # Relax the fleet budget and use the full [-1,1] predictor range:
        # a removed bus cannot change either training safety event.
        low += (midpoint - np.abs(drift)).min(axis=0)
        high += (midpoint + np.abs(drift)).max(axis=0)
    critical = (low < .952) | (high > 1.048)
    monitor_ids = np.asarray(monitors) - 1
    active = critical.copy()
    active[monitor_ids] = True
    bus_ids = np.flatnonzero(active)
    variable = np.any(np.concatenate(variations, axis=0) != 0, axis=0) & active
    varying_cols = np.flatnonzero(variable[bus_ids])
    reduced = [replace(s, voltage_shifts=np.column_stack((m[:, active], d[:, variable])))
               for s, m, d in zip(stations, offsets, variations, strict=True)]
    monitor_cols = np.searchsorted(bus_ids, monitor_ids)
    at_risk = critical[bus_ids]
    values = v[:, bus_ids] - 1.
    scaled = normalized[:, bus_ids]
    model_loss = model.operator.shift_evaluator(mixture['weights'],
                                               mixture['local_probabilities'], evidence)
    prior_weight = 0. if snapshot_proposal else 32 / (32 + matches)
    bands = (.048, .05)
    original = [np.any(np.abs(values) > b + 1e-12, axis=1) for b in bands]
    def risk(boundaries):
        means = boundaries[:, :len(bus_ids)]
        prior = model_loss(means[:, None, monitor_cols])[:, 0, :].mean(axis=1)
        result = np.empty(len(boundaries))
        # Bounded working memory, including for unobserved, full-bucket contexts.
        batch = max(1, min(96, 160_000//len(values)))
        for start in range(0, len(boundaries), batch):
            m = means[start:start+batch]
            d = np.zeros_like(m)
            d[:, varying_cols] = boundaries[start:start+len(m), len(bus_ids):]
            shape = (len(m), len(values))
            events = [np.zeros(shape, bool) for _ in bands]
            peak_excess = np.zeros(shape)
            endpoint = np.zeros(len(m))
            for j in range(len(bus_ids)):
                deviation = np.abs(values[None, :, j] + m[:, j, None]
                                   + d[:, j, None]*scaled[None, :, j])
                if j in monitor_cols:
                    endpoint += np.maximum(deviation-DEADBAND, 0.).mean(axis=1)/len(monitor_cols)
                if at_risk[j]:
                    for event, band in zip(events, bands, strict=True):
                        event |= deviation > band + 1e-12
                    np.maximum(peak_excess, deviation-.05, out=peak_excess)
            acceptable = np.ones(len(m), bool)
            for event, baseline in zip(events, original, strict=True):
                new = (event & ~baseline).sum(axis=1)
                rescued = (~event & baseline).sum(axis=1)
                acceptable &= (new == 0) | (new-rescued + 1.645*np.sqrt(new+rescued) <= 0)
            # One p.u. of worst-bus operational excess has the same units as
            # monitor deadband loss. Use an explicit, fixed relative weight.
            scores = (prior_weight*prior[start:start+len(m)] + (1-prior_weight)*endpoint
                      + .25*peak_excess.mean(axis=1))
            result[start:start+len(m)] = np.where(acceptable, scores, np.inf)
        return result

    result = solve_dispatch(reduced, risk, tick=tick, resource_limits=BUDGET, transport=transport)
    shift = center - no_action_voltages[rows].mean(axis=0)
    for m, choice in zip(offsets, result.action_indices, strict=True):
        shift += m[choice]
    diagnostics = dict(result.diagnostics, response_channels=len(bus_ids)+len(varying_cols),
                       protected_buses=int(critical.sum()), matched_training_rows=matches,
                       training_rows=len(rows), prior_weight=prior_weight,
                       response_rounding_bound=len(stations)*tick)
    return replace(result, voltage_shift=tuple(map(float, shift)), diagnostics=diagnostics)


@dataclass
class QualifiedPolicies:
    actions: tuple
    shifts: np.ndarray
    states: np.ndarray
    buckets: np.ndarray
    losses: np.ndarray
    excursions: np.ndarray
    source_rows: np.ndarray


def build_qualification(controller, feeder, powers, timestamps, observed, capacity=48):
    """Discover a bounded set with distributed sweeps; replay it on later training days.

    Shared categorization, mixture fits and response calibration use the training
    period. Proposal context frequencies and single-snapshot probes use its early
    days. Qualification uses only the later days' exact AC outcomes.
    """
    days = np.asarray(timestamps) // 48
    distinct_days = np.unique(days)
    if len(distinct_days) < 32:
        return None
    early = days < distinct_days[int(.6*len(distinct_days))]
    later = np.flatnonzero(~early)
    train = replace(controller, training_states=controller.training_states[early],
                    training_buckets=controller.training_buckets[early],
                    reference_voltages={k: v[early] for k, v in controller.reference_voltages.items()})
    presets = {}
    observed = tuple(i for i in observed if 0 <= i < len(train.site_buses))
    grid = train.stations[-1]
    lower = min(q for _, q in grid.actions)
    limits = ({}, {grid.bus: (lower, 0.)}) if lower <= 0 else ({},)
    no_action = ACTIONS.index((0., 0.))
    for bucket in range(4):
        selected = np.flatnonzero(train.training_buckets == bucket)
        if not len(selected):
            continue
        contexts = [{}]
        if observed:
            patterns, counts = np.unique(train.training_states[selected][:, observed],
                                         axis=0, return_counts=True)
            for i in np.argsort(-counts, kind='stable')[:6]:
                contexts.append(dict(zip((train.site_buses[j] for j in observed),
                                         map(int, patterns[i]), strict=True)))
        for constraint in limits:
            for context in contexts:
                stations, indices, evidence = train._inputs(bucket, context, constraint)
                ref, _ = train._reference(stations, bucket, evidence)
                rows, matches = select_training_context(train.training_states, train.training_buckets,
                                                       bucket, evidence)
                legacy = train.dispatch(bucket, context, q_limits=constraint, method="legacy")
                presets.setdefault(legacy.actions, None)
                response = train.responses[ref]
                base = train.reference_voltages[ref]
                args = (train.mixtures[bucket], evidence, train.monitor_buses, train.tick, 'local')
                proposal = adaptive_dispatch(stations, indices, response, base,
                                             train.reference_voltages[no_action], rows, matches, *args)
                presets.setdefault(proposal.actions, None)
                # Tail operating points generate proposals for asymmetric support
                # and high/low extremes, even when the categorical mean is benign.
                if not context:
                    monitored = base[rows][:, np.asarray(train.monitor_buses)-1]
                    for col in monitored.T:
                        order = rows[np.argsort(col)]
                        for i in (len(rows)//10, 9*len(rows)//10):
                            probe = adaptive_dispatch(stations, indices, response, base,
                                                      train.reference_voltages[no_action], order[i:i+1],
                                                      1, *args, snapshot_proposal=True)
                            presets.setdefault(probe.actions, None)
    actions = tuple(presets)[:capacity]
    if not actions:
        return None
    shifts, losses, excursions = [], [], []
    monitor_ids = np.asarray(controller.monitor_buses) - 1
    for schedule in actions:
        injections = [(station.bus, *pq) for station, pq in zip(controller.stations, schedule, strict=True)]
        v = np.abs(feeder.ac_batch(powers[later], actions=injections))
        losses.append(np.maximum(np.abs(v[:, monitor_ids]-1.) - DEADBAND, 0.).mean(axis=1))
        excursions.append(np.max(np.abs(v-1.), axis=1))
        delta = np.zeros(v.shape[1])
        for s, pq in zip(controller.stations, schedule, strict=True):
            i = list(map(tuple, s.actions)).index(pq)
            delta += np.rint(s.voltage_shifts[i]/controller.tick)*controller.tick
        shifts.append(delta)
    return QualifiedPolicies(actions, np.asarray(shifts), controller.training_states[later],
                             controller.training_buckets[later], np.asarray(losses),
                             np.asarray(excursions), later)


def qualify_dispatch(controller, incumbent, stations, bucket, evidence, transport):
    """Compare paired historical AC outcomes and enact an accepted policy by neighbor agreement."""
    archive = controller.qualification
    info = dict(proposal_count=0 if archive is None else len(archive.actions),
                qualification_rows=0, selected_policy="incumbent", agreement_sweeps=0)
    def unchanged():
        return replace(incumbent, diagnostics=dict(incumbent.diagnostics, **info))
    if archive is None or incumbent.actions not in archive.actions:
        return unchanged()
    rows = np.flatnonzero(archive.buckets == bucket)
    for site, level in evidence.items():
        rows = rows[archive.states[rows, site] == level]
    info["qualification_rows"] = len(rows)
    if len(rows) < 16:
        return unchanged()
    incumbent_id = archive.actions.index(incumbent.actions)
    # Every archived schedule was produced by a budget-feasible neighbor sweep.
    # The same continuous source records are used for both actions of every pair.
    history = archive.losses[:, rows]
    diffs = history - history[incumbent_id]
    excess = archive.excursions[:, rows]
    ref = archive.excursions[incumbent_id, rows]
    confidence = 1.645 * diffs.std(axis=1, ddof=1) / np.sqrt(len(rows))
    eligible = diffs.mean(axis=1) + confidence < -1e-12
    for band in (.048, .05):
        bad = excess > band + 1e-12
        baseline_bad = ref > band + 1e-12
        new = (bad & ~baseline_bad).sum(axis=1)
        saved = (~bad & baseline_bad).sum(axis=1)
        eligible &= (new == 0) | (new-saved + 1.645*np.sqrt(new+saved) <= 0)
    if not eligible.any():
        return unchanged()
    mix = controller.mixtures[bucket]
    evaluate = controller.shift_operator.shift_evaluator(mix['weights'], mix['local_probabilities'], evidence)
    monitor_ids = np.asarray(controller.monitor_buses)-1
    prior = evaluate(archive.shifts[:, None, monitor_ids])[:, 0, :].mean(axis=1)
    mix_weight = 32/(32+len(rows))
    score = (mix_weight*prior + (1-mix_weight)*history.mean(axis=1)
             + .25*np.maximum(excess-.05, 0.).mean(axis=1))
    base_score = float(score[incumbent_id])
    eligible &= score < base_score - 1e-12
    if not eligible.any():
        return unchanged()
    from policy_agreement import agree_archive
    eligible[incumbent_id] = True
    scores = np.where(eligible, score, np.inf)
    winner, indices, agreement = agree_archive(stations, archive.actions, scores, transport=transport)
    info.update(selected_policy="incumbent" if winner == incumbent_id else "qualified",
                agreement_sweeps=1, prior_weight=mix_weight,
                paired_endpoint_improvement=float(-diffs[winner].mean()))
    diag = dict(incumbent.diagnostics, **info)
    for key in ('messages', 'forward_messages', 'backward_messages', 'wire_bytes', 'raw_wire_bytes'):
        if key in agreement:
            diag[key] += agreement[key]
    diag['agreement_mask_bytes'] = agreement['archive_mask_bytes']
    diag['agreement_id_bytes'] = agreement['policy_id_bytes']
    usage = sum(np.asarray(s.resources)[i] for s, i in zip(stations, indices, strict=True))
    diag['resources_used'] = tuple(map(int, usage))
    return replace(incumbent, action_indices=tuple(indices), actions=archive.actions[winner],
                   voltage_shift=tuple(archive.shifts[winner]), objective=float(score[winner]),
                   diagnostics=diag)
