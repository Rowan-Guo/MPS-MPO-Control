"""Non-enumerative correlated Born MPS and physics-derived loss MPO.

The high-dimensional experiment uses first-order LinDistFlow voltage sensitivity,
NOT the nonlinear AC loss tensor of the 10-bus experiment. The non-smooth
voltage deadband hinge is represented by a truncated *cosine Fourier series*.
Its TT rank is at most 2*harmonics+1, independent of the number of loads.
Every approximation and its uniform error bound is explicit.
"""
from __future__ import annotations

from itertools import product
from math import pi
import numpy as np

from powerflow import LEVELS, ACTIONS, loss
from tensor import contract, identity_mpo


def correlated_born_mps(local_probabilities, amplitudes):
    """Positive, normalized mixture-of-product *amplitudes*, NOT probabilities.

    psi(x) = C sum_k amplitudes[k] product_i sqrt(p[k,i,x_i]).
    The Born density p(x) = |psi(x)|^2 is correlated. Its MPS rank <= K.
    Construction never creates the 3**m table. This is a KNOWN distribution,
    not one learned from finite training samples.
    """
    p = np.asarray(local_probabilities, dtype=float)
    w = np.asarray(amplitudes, dtype=float)
    if (p.ndim != 3 or p.shape[1] < 2 or p.shape[2] < 2 or
        not np.all(np.isfinite(p)) or np.any(p < 0) or
        not np.allclose(p.sum(axis=-1), 1, atol=1e-12) or
        w.shape != (p.shape[0],) or not np.all(np.isfinite(w)) or
        np.any(w < 0) or w.sum() <= 0):
        raise ValueError('expected K x sites x categories probability distributions and positive mixture weights')
    k, sites, dim = p.shape
    roots = np.sqrt(p)
    if sites == 1:
        state = [np.einsum('k,kd->d', w, roots[:, 0, :]).reshape(1, dim, 1)]
    else:
        state = [np.einsum('k,kd->dk', w, roots[:, 0, :]).reshape(1, dim, k)]
        for site in range(1, sites - 1):
            core = np.zeros((k, dim, k))
            indices = np.arange(k)
            core[indices, :, indices] = roots[:, site, :]
            state.append(core)
        state.append(roots[:, -1, :].reshape(k, dim, 1))
    squared_norm = contract(state, identity_mpo(state))
    state[0] /= np.sqrt(squared_norm)
    return tuple(state)


def demo_correlations(sites):
    """Two latent load regimes with smooth site variation; no dense enumeration."""
    if sites < 2:
        raise ValueError('the experiment requires at least two load sites')
    p = np.empty((2, sites, 3))
    for i in range(sites):
        t = 0.05 * np.sin(i * 0.61)
        p[0, i] = [0.70 + t, 0.23 - t, 0.07]
        p[1, i] = [0.08, 0.24 + t, 0.68 - t]
    return p, np.array([0.55, 0.45])


def sample_born_mixture(local_probabilities, amplitudes, seen, n, rng):
    """Sample the *Born* distribution conditional on observations, without TT
    reconstruction or Markov-chain mixing. A positive K-component amplitude
    induces a positive K^2-component mixture of product probabilities.
    """
    p = np.asarray(local_probabilities, dtype=float)
    w = np.asarray(amplitudes, dtype=float)
    k, sites, dim = p.shape
    if n < 1:
        raise ValueError('n must be positive')
    for i, v in seen.items():
        if not (0 <= i < sites and 0 <= v < dim):
            raise ValueError('invalid evidence')
    sqrt = np.sqrt(p)
    pairs = tuple(product(range(k), repeat=2))
    weights = np.empty(len(pairs))
    components = np.empty((len(pairs), sites, dim))
    for h, (a, b) in enumerate(pairs):
        overlap_terms = sqrt[a] * sqrt[b]
        overlaps = overlap_terms.sum(axis=-1)
        components[h] = overlap_terms / overlaps[:, None]
        weight = w[a] * w[b] * np.prod(overlaps)
        for i, v in seen.items():
            weight *= components[h, i, v]
        weights[h] = weight
    if weights.sum() <= 0:
        raise ValueError('zero probability evidence')
    choice = rng.choice(len(pairs), size=n, p=weights / weights.sum())
    out = np.empty((n, sites), dtype=int)
    for i in range(sites):
        if i in seen:
            out[:, i] = seen[i]
        else:
            # Vectorized independent draws conditional on the chosen pair.
            cdf = np.cumsum(components[choice, i], axis=-1)
            out[:, i] = np.sum(rng.random(n)[:, None] > cdf[:, :-1], axis=1)
    return out


def fourier_coefficients(deadband, domain, harmonics):
    """f(x)=max(0,|x|-db) on [-domain,domain], cosine Fourier series.

    The periodic extension is only a sound surrogate if all reachable x are
    within that domain. Absolute tail bound: 4*domain/(pi*pi*harmonics).
    """
    if not (0 <= deadband < domain) or harmonics < 1:
        raise ValueError('require 0 <= deadband < domain and harmonics >= 1')
    k = np.arange(1, harmonics + 1)
    a0half = (domain - deadband)**2 / (2 * domain)
    ak = 2 * domain * ((-1.) ** k - np.cos(k * pi * deadband / domain)) / (pi*k)**2
    return a0half, ak


def fourier_uniform_error_bound(domain, harmonics):
    """Guaranteed sup-norm bound for the periodic deadband approximation.

    Uses |a_k| <= 4D/(pi^2*k^2), and sum_{k>N} 1/k^2 <= 1/N.
    This is a conservative representation bound, NOT an AC modeling bound.
    """
    return 4 * domain / (pi**2 * harmonics)


def fourier_deadband_mpo(constant, contributions, deadband, domain, harmonics):
    """Diagonal TT for Fourier-approximated deadband loss of c+sum_i f_i(x_i).

    Independent of 3**m; per-site TT bond R = 1 + 2*harmonics. All
    contractions are real and stable (sine/cosine rotations, no power series).
    """
    f = np.asarray(contributions, dtype=float)
    if f.ndim != 2 or f.shape[0] < 1 or f.shape[1] < 2:
        raise ValueError('contributions must be (sites, categories)')
    lower = constant + f.min(axis=1).sum()
    upper = constant + f.max(axis=1).sum()
    if max(abs(lower), abs(upper)) > domain + 1e-12:
        raise ValueError('domain fails to enclose all possible voltage deviations')
    a0, ak = fourier_coefficients(deadband, domain, harmonics)
    sites, dim = f.shape
    k = np.arange(1, harmonics + 1)
    freq = pi * k / domain
    if sites == 1:
        vals = a0 + np.sum(ak[:, None] * np.cos(freq[:, None] * (constant + f[0])), axis=0)
        return (vals.reshape(1, dim, 1),)
    rank = 1 + 2 * harmonics
    first = np.zeros((1, dim, rank))
    first[0, :, 0] = 1.
    for h, omega in enumerate(freq):
        angle = omega * (constant + f[0])
        first[0, :, 1 + 2*h] = np.cos(angle)
        first[0, :, 2 + 2*h] = np.sin(angle)
    cores = [first]
    for site in range(1, sites - 1):
        block = np.zeros((rank, dim, rank))
        block[0, :, 0] = 1.
        for h, omega in enumerate(freq):
            ang = omega * f[site]
            cs, sn = np.cos(ang), np.sin(ang)
            ix = 1 + 2*h
            block[ix, :, ix] = cs
            block[ix, :, ix + 1] = sn
            block[ix + 1, :, ix] = -sn
            block[ix + 1, :, ix + 1] = cs
        cores.append(block)
    last = np.zeros((rank, dim, 1))
    last[0, :, 0] = a0
    for h, omega in enumerate(freq):
        ang = omega * f[-1]
        ix = 1 + 2*h
        last[ix, :, 0] = ak[h] * np.cos(ang)
        last[ix + 1, :, 0] = -ak[h] * np.sin(ang)
    cores.append(last)
    return tuple(cores)


def feeder_spec(sites):
    """An m+4-bus radial chain. Scale each line impedance for size comparability.

    This synthetic feeder is NOT a named IEEE benchmark. One uncertain load
    at buses 2..m+1, deterministic 0.02 at other non-slack buses; an inverter
    at the terminal bus. Bus N-1 voltage linearization is around 1 p.u.
    """
    buses = sites + 4
    r = 0.01 * (9 / (buses - 1))**2
    x = 2 * r
    locations = np.arange(2, 2 + sites)
    all_nodes = np.arange(1, buses)
    fixed_nodes = np.setdiff1d(all_nodes, locations)
    fixed_weight = 0.02 * fixed_nodes.sum()
    nominal_weight = 0.10 * locations.sum() + fixed_weight
    slack = 1.0 + r * nominal_weight - 0.006
    return dict(buses=buses, r=r, x=x, locations=locations,
                fixed_nodes=fixed_nodes, fixed_weight=fixed_weight, slack=slack)


def terminal_model(sites, action, spec=None):
    """Return delta V_end = constant + sum_i contrib[i, category].

    First-order radial voltage linearization with constant-PQ loads. Action
    injects real and reactive power at the final bus. No nonlinear AC claim.
    """
    if spec is None:
        spec = feeder_spec(sites)
    c = (spec['slack'] - 1 - spec['r'] * spec['fixed_weight'] / spec['slack']
         + (spec['buses']-1) * (spec['r'] * action[0] + spec['x'] * action[1]) / spec['slack'])
    f = (-spec['r'] / spec['slack'] * spec['locations'][:, None]
         * np.asarray(LEVELS)[None, :])
    return c, f


def ac_terminal_voltages(p_values, action, spec):
    """AC sweep of one arbitrary continuous load vector, offline only."""
    load = np.zeros(spec['buses'], dtype=complex)
    load[1:] = 0.02
    load[spec['locations']] = np.asarray(p_values, dtype=float)
    load[-1] -= complex(*action)
    z = spec['r'] + 1j * spec['x']
    voltages = np.full(spec['buses'], complex(spec['slack']))
    for _ in range(150):
        currents = np.zeros(spec['buses'] + 1, dtype=complex)
        for j in range(spec['buses'] - 1, 0, -1):
            currents[j] = (load[j] / voltages[j]).conjugate() + currents[j + 1]
        next_voltages = np.empty_like(voltages)
        next_voltages[0] = spec['slack']
        for j in range(1, spec['buses']):
            next_voltages[j] = next_voltages[j-1] - z * currents[j]
        if np.max(np.abs(next_voltages-voltages)) < 1e-12:
            return abs(next_voltages)
        voltages = next_voltages
    raise RuntimeError('high-dimensional AC validation failed to converge')


def full_ac(states, action, spec):
    """Independent nonlinear AC validation, not used to evaluate online scores.

    Rows are categorical states, not continuous load magnitudes.
    """
    out = np.empty(len(states))
    unsafe = np.zeros(len(states), dtype=bool)
    for row, state in enumerate(states):
        mags = ac_terminal_voltages(np.asarray(LEVELS)[state], action, spec)
        out[row] = mags[-1]
        unsafe[row] = mags.min() < 0.95 or mags.max() > 1.05
    return out, unsafe


def ac_calibrated_terminal_models(sites, actions, spec=None, *,
                                  reference_power=0.10, difference_step=0.02):
    """Non-enumerative AC-informed linearization: O(actions * sites) sweeps.

    Finite differences at a shared physical reference estimate each voltage
    magnitude sensitivity, separately for every action. These are local affine
    surrogates, NOT certified approximations to the nonlinear AC voltage.
    """
    if spec is None:
        spec = feeder_spec(sites)
    if not 0 < difference_step < reference_power:
        raise ValueError('invalid finite-difference step')
    reference = np.full(sites, reference_power, dtype=float)
    models = []
    for action in actions:
        voltage_ref = ac_terminal_voltages(reference, action, spec)[-1]
        derivatives = np.empty(sites)
        for j in range(sites):
            plus, minus = reference.copy(), reference.copy()
            plus[j] += difference_step
            minus[j] -= difference_step
            derivatives[j] = (ac_terminal_voltages(plus, action, spec)[-1] -
                              ac_terminal_voltages(minus, action, spec)[-1]) / (2*difference_step)
        constant = voltage_ref - 1 - derivatives @ reference
        contributions = derivatives[:, None] * np.asarray(LEVELS)[None, :]
        models.append((constant, contributions))
    return tuple(models), len(actions)*(2*sites+1)


def direct_deadband_scores(states, actions, spec, deadband, *, models=None):
    """Direct affine-model deadband losses for Monte Carlo or small-N checks."""
    if models is None:
        models = tuple(terminal_model(states.shape[1], a, spec) for a in actions)
    return np.stack([np.maximum(0, np.abs(c + f[np.arange(len(f))[None, :], states].sum(axis=1)) - deadband)
                     for c, f in models], axis=-1)
