"""Lag-conditioned spatial mixture TT and training-record selection for control."""
from dataclasses import dataclass, replace

import numpy as np


LAGS = (1, 48)


def _posterior(mixture, evidence):
    mass = np.log(np.maximum(mixture['weights'], 1e-300))
    for site, category in evidence.items():
        mass = mass + np.log(np.maximum(mixture['local_probabilities'][:, site, category], 1e-300))
    mass = np.exp(mass - np.max(mass))
    return mass / mass.sum()


def category_marginals(mixture, evidence):
    posterior = _posterior(mixture, evidence)
    marginals = np.einsum('c,cid->id', posterior, mixture['local_probabilities'])
    for site, category in evidence.items():
        marginals[site] = 0.
        marginals[site, category] = 1.
    return marginals


@dataclass
class TemporalModel:
    # lag -> per-training-row index of its observed predecessor, or -1 for a gap
    predecessors: dict
    # (lag, previous clock, current clock) -> latent transition, local emissions, pair count
    transitions: dict

    @classmethod
    def fit(cls, states, buckets, timestamps, mixtures):
        times = np.asarray(timestamps)
        if (times.ndim != 1 or times.dtype.kind not in 'iu'
                or len(np.unique(times)) != len(times)):
            raise ValueError("training half-hour timestamps must be distinct integers")
        row_at = {int(t): i for i, t in enumerate(times)}
        responsibilities = np.empty((len(states), len(mixtures[0]['weights'])))
        for k, mix in enumerate(mixtures):
            rows = np.flatnonzero(buckets == k)
            shape = (len(rows), responsibilities.shape[1])
            logs = np.broadcast_to(np.log(np.maximum(mix['weights'], 1e-300)), shape).copy()
            for site in range(states.shape[1]):
                p = mix['local_probabilities'][:, site, :]
                logs += np.log(np.maximum(p[:, states[rows, site]].T, 1e-300))
            prob = np.exp(logs-logs.max(axis=1, keepdims=True))
            responsibilities[rows] = prob/prob.sum(axis=1, keepdims=True)
        predecessors, transitions = {}, {}
        for lag in LAGS:
            prev = np.array([row_at.get(int(t)-lag, -1) for t in times], dtype=int)
            predecessors[lag] = prev
            for current in range(4):
                for past in range(4):
                    rows = np.flatnonzero((prev >= 0) & (buckets == current) & (buckets[prev] == past))
                    if len(rows) < 16:
                        continue
                    joint = responsibilities[prev[rows]].T @ responsibilities[rows]
                    # Shrink towards independent clock-conditioned components.
                    joint += 32*np.outer(mixtures[past]['weights'], mixtures[current]['weights'])
                    # Given a current component and a site's previous category,
                    # estimate the site's next emission with 32 pseudocounts
                    # shared across components according to their prior mass.
                    local = mixtures[current]['local_probabilities']
                    counts = np.empty((*local.shape, 3))
                    for site in range(states.shape[1]):
                        pair_code = 3*states[prev[rows], site] + states[rows, site]
                        for component, mass in enumerate(mixtures[current]['weights']):
                            table = np.bincount(pair_code, weights=responsibilities[rows, component],
                                                minlength=9).reshape(3, 3)
                            table += 32*mass*local[component, site]
                            counts[component, site] = table / table.sum(axis=1, keepdims=True)
                    transitions[lag, past, current] = (joint / joint.sum(axis=1, keepdims=True),
                                                        counts, len(rows))
        return cls(predecessors, transitions)

    def infer(self, mixtures, sites, bucket, history, half_hour):
        if half_hour is not None and (not isinstance(half_hour, (int, np.integer))
                                      or (half_hour % 48)//12 != bucket):
            raise ValueError("half_hour must be an integer consistent with the clock bucket")
        base = mixtures[bucket]['weights']
        predictions, lag_evidence = [], {}
        base_local = mixtures[bucket]['local_probabilities']
        local_sum = base_local.copy()
        local_count = np.ones(len(sites))
        for lag, reports in (history or {}).items():
            if lag not in LAGS:
                raise ValueError("supported historical lags are 1 and 48 half-hours")
            if not reports:
                continue
            if any(b not in sites or not isinstance(level, (int, np.integer)) or level not in range(3)
                   for b, level in reports.items()):
                raise ValueError("historical reports require configured site IDs and categories 0..2")
            evidence = {sites.index(bus): int(level) for bus, level in reports.items()}
            # If only a coarse clock bucket is available, average over previous
            # clock buckets with their empirical frequencies from training pairs.
            clocks = range(4) if half_hour is None else (((half_hour-lag) % 48)//12,)
            prediction = np.zeros_like(base)
            emissions = np.zeros_like(base_local)
            total = 0
            for past in clocks:
                pair = self.transitions.get((lag, past, bucket))
                if pair is None:
                    continue
                transition, local, count = pair
                prev = mixtures[past]
                delta = (_posterior(prev, evidence)-prev['weights']) @ transition
                prediction += count*delta
                for site, category in evidence.items():
                    emissions[:, site] += count*local[:, site, category]
                total += count
            if total:
                lag_evidence[lag] = evidence
                predictions.append(prediction / total)
                for site in evidence:
                    local_sum[:, site] += emissions[:, site]/total
                    local_count[site] += 1
        if not predictions:
            return mixtures[bucket], {}
        # Each lag supplies an alternative predictive prior for the same time.
        # Averaging avoids double counting correlated past reports. Keep one
        # spatial prior's weight alongside the available predictive priors.
        reweighted = np.maximum(base + sum(predictions)/(len(predictions)+1), 1e-12)
        return dict(mixtures[bucket], weights=reweighted/reweighted.sum(),
                    local_probabilities=local_sum/local_count[None, :, None]), lag_evidence

    def conditioned_rows(self, states, buckets, bucket, evidence, lag_reports, source_rows=None):
        """Pool matching past/current records with up to 64 evenly spaced spatial records.

        Drop the daily lag before dropping the recent report when their joint
        empirical support is small. Every included row is a distinct training
        interval. No missing predecessor is filled across gaps.
        """
        rows = np.flatnonzero(buckets == bucket)
        if source_rows is not None:
            rows = source_rows[buckets[source_rows] == bucket]
        for site, category in evidence.items():
            rows = rows[states[rows, site] == category]
        if len(rows) < 32:
            return None
        accepted = {}
        for lag, reports in lag_reports.items():
            prev = self.predecessors[lag][rows]
            valid = prev >= 0
            for site, category in reports.items():
                valid &= states[prev, site] == category
            accepted[lag] = valid
        ordered = [accepted[k] for k in LAGS if k in accepted]
        if not ordered:
            return None
        for mask in [np.logical_and.reduce(ordered), *ordered]:
            if np.sum(mask) < 32 or np.all(mask):
                continue
            other = rows[~mask]
            # Smooth sparse conditional curves toward the spatial curve while
            # keeping sample sizes honest for the paired historical gate.
            count = min(len(other), round(64*len(other)/len(rows)))
            background = other[np.linspace(0, len(other)-1, count, dtype=int)]
            return np.sort(np.concatenate((rows[mask], background)))
        return None


def with_history(controller, bucket, current_evidence, history, half_hour):
    mixture, reports = controller.temporal.infer(controller.mixtures, controller.site_buses,
                                                bucket, history, half_hour)
    if not reports:
        return controller
    mixes = list(controller.mixtures)
    mixes[bucket] = mixture
    adjusted = replace(controller, mixtures=mixes)
    select = controller.temporal.conditioned_rows
    inputs = controller.training_states, controller.training_buckets, bucket, current_evidence, reports
    rows = select(*inputs)
    if rows is not None:
        adjusted.training_states = controller.training_states[rows]
        adjusted.training_buckets = controller.training_buckets[rows]
        adjusted.reference_voltages = {i: v[rows] for i, v in controller.reference_voltages.items()}
    archive = controller.qualification
    if archive is not None:
        kept = select(*inputs, source_rows=archive.source_rows)
        if kept is not None:
            local = np.searchsorted(archive.source_rows, kept)
            adjusted.qualification = replace(archive, states=archive.states[local], buckets=archive.buckets[local],
                                             losses=archive.losses[:, local], excursions=archive.excursions[:, local],
                                             source_rows=kept)
    return adjusted
