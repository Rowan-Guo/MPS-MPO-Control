"""Classical tensor-train (MPS) Born probabilities and diagonal MPO contractions.

Each categorical load is ONE physical site, with its own local dimension (normally 3).
There are no padded binary states. The diagonal MPO is stored as its diagonal
TT slices of shape (r_left, d, r_right), not a dense four-index operator.

For a nonnegative normalized joint table p, ``amplitude_state(p)`` gives an MPS
representing sqrt(p). Rank truncation is performed on the *amplitude*; the
resulting Born distribution is re-normalized and remains nonnegative. A truncated
operator is only an approximation; safety indicators should not be truncated.
"""
from __future__ import annotations

from itertools import product
import numpy as np

STATES = tuple(product(range(3), repeat=3))  # Legacy 10-bus demonstration.


def tt_svd(values, *, max_bond=None, relative_tolerance=0.0):
    """TT-SVD of a dense real tensor with optional rank cap and Frobenius budget.

    For relative_tolerance>0 the per-cut squared discarded singular-value sum
    is bounded by ``(relative_tolerance * ||values||_F)^2/(n_sites-1)`` *unless*
    max_bond requires a smaller rank. The rank cap can invalidate this guarantee.
    Constructing ``values`` itself is exponential in the number of sites;
    analytical TT builders are needed for high-dimensional non-enumerative use.
    """
    dense = np.asarray(values, dtype=float)
    if dense.ndim < 1 or any(dim < 2 for dim in dense.shape):
        raise ValueError('expected one or more categorical dimensions >= 2')
    if not np.all(np.isfinite(dense)):
        raise ValueError('TT values must be finite')
    if max_bond is not None and (not isinstance(max_bond, (int, np.integer)) or max_bond < 1):
        raise ValueError('max_bond must be a positive integer or None')
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError('relative_tolerance must be nonnegative and finite')
    n = dense.ndim
    if n == 1:
        return (dense.reshape(1, dense.size, 1).copy(),)
    allowed_sq = (relative_tolerance * np.linalg.norm(dense.ravel())) ** 2 / (n - 1)
    out = []
    work = dense.copy()
    bond = 1
    for site in range(n - 1):
        dimension = dense.shape[site]
        u, s, vh = np.linalg.svd(work.reshape(bond * dimension, -1), full_matrices=False)
        # Relative, scale-invariant threshold for exact reconstructions.
        numerical_rank = max(1, int(np.count_nonzero(s > s[0] * 1e-13)))
        target_rank = numerical_rank
        if relative_tolerance > 0:
            discarded = np.cumsum((s[:numerical_rank][::-1] ** 2))[::-1]
            for r in range(1, numerical_rank):
                if discarded[r] <= allowed_sq:
                    target_rank = r
                    break
        if max_bond is not None:
            target_rank = min(target_rank, max_bond)
        out.append(u[:, :target_rank].reshape(bond, dimension, target_rank))
        work = s[:target_rank, None] * vh[:target_rank]
        bond = target_rank
    out.append(work.reshape(bond, dense.shape[-1], 1))
    return tuple(out)


def validate_cores(cores, *, dims=None):
    if not cores:
        raise ValueError('at least one tensor core is required')
    prev = 1
    dimensions = []
    for core in cores:
        if core.ndim != 3 or core.shape[0] != prev or core.shape[1] < 2:
            raise ValueError('inconsistent TT core bond or physical dimensions')
        prev = core.shape[-1]
        dimensions.append(core.shape[1])
    if prev != 1:
        raise ValueError('right boundary bond must be one')
    if dims is not None and tuple(dimensions) != tuple(dims):
        raise ValueError('TT physical dimensions mismatch')
    return tuple(dimensions)


def identity_mpo(psi):
    return tuple(np.ones((1, core.shape[1], 1)) for core in psi)


def evidence(select, dims=(3, 3, 3)):
    """Site -> categorical index. Each index is observed as one local projector."""
    if any(int(d) != d or d < 2 for d in dims):
        raise ValueError('invalid categorical dimensions')
    mask = [np.ones(int(dim)) for dim in dims]
    for site, value in select.items():
        if not isinstance(site, (int, np.integer)) or not (0 <= site < len(dims)):
            raise ValueError('invalid plant/site index')
        if not isinstance(value, (int, np.integer)) or not (0 <= value < dims[site]):
            raise ValueError('invalid observation category')
        mask[site][:] = 0.0
        mask[site][value] = 1.0
    return tuple(mask)


def contract(psi, op, mask=None):
    """Compute <psi| diag(op) diag(mask) |psi> without dense reconstruction."""
    dims = validate_cores(psi)
    validate_cores(op, dims=dims)
    if len(op) != len(psi):
        raise ValueError('state/operator number of sites differs')
    if mask is None:
        mask = tuple(np.ones(d) for d in dims)
    if len(mask) != len(dims) or any(np.shape(m) != (d,) for m, d in zip(mask, dims)):
        raise ValueError('invalid evidence mask')
    environment = np.ones((1, 1, 1))
    for a, w, m in zip(psi, op, mask, strict=True):
        environment = np.einsum('ijk,isa,jsb,ksc,s->abc', environment,
                                a.conj(), w, a, m, optimize=True)
    return float(np.real_if_close(environment[0, 0, 0]))


def amplitude_state(joint, *, max_bond=None, relative_tolerance=0.0):
    """Construct a normalized Born MPS from a small dense probability table."""
    joint = np.asarray(joint, dtype=float)
    if (joint.ndim == 0 or not np.all(np.isfinite(joint)) or np.any(joint < 0)
            or not np.isclose(joint.sum(), 1, atol=1e-10, rtol=0)):
        raise ValueError('expected a nonnegative normalized joint probability table')
    psi = list(tt_svd(np.sqrt(joint), max_bond=max_bond,
                      relative_tolerance=relative_tolerance))
    squared_norm = contract(psi, identity_mpo(psi))
    if squared_norm <= 0:
        raise ValueError('degenerate truncated probability amplitude')
    psi[0] = psi[0] / np.sqrt(squared_norm)
    return tuple(psi)


def diagonal_mpo(values, *, max_bond=None, relative_tolerance=0.0):
    """Diagonal operator (loss or unsafe indicator) via dense TT-SVD (small N)."""
    return tt_svd(values, max_bond=max_bond, relative_tolerance=relative_tolerance)


def conditional_scores(psi, ops, seen, unsafe=None, *, risk_tolerance=0.0,
                       return_risks=False):
    """Expected losses and probability of evidence under a Born MPS.

    If unsafe operators are supplied, actions with conditional violation
    probability > risk_tolerance are discarded. For 0 the 1e-10 numerical
    guard only ignores floating-point cancellation, not physical violations.
    Rank-truncated Born states reflect an *approximated* probability model.
    """
    if not 0 <= risk_tolerance <= 1:
        raise ValueError('risk_tolerance must be in [0, 1]')
    dims = validate_cores(psi)
    mask = evidence(seen, dims)
    den = contract(psi, identity_mpo(psi), mask)
    if den <= 1e-14:
        raise ValueError('zero probability evidence')
    if unsafe is not None and len(unsafe) != len(ops):
        raise ValueError('one safety operator per action is required')
    risks = (np.zeros(len(ops)) if unsafe is None else
             np.array([np.clip(contract(psi, flag, mask) / den, 0, 1)
                       for flag in unsafe]))
    # Only a numerical tolerance is allowed; operational tolerance is explicit.
    valid = risks <= risk_tolerance + 1e-10
    scores = np.array([contract(psi, operator, mask) / den if ok else np.inf
                       for operator, ok in zip(ops, valid, strict=True)])
    return (scores, den, risks) if return_risks else (scores, den)


def posterior(psi, seen, missing, den=None):
    """Conditional marginal on one unobserved categorical site."""
    dims = validate_cores(psi)
    if missing in seen or not 0 <= missing < len(dims):
        raise ValueError('posterior target must be an unobserved site')
    if den is None:
        den = contract(psi, identity_mpo(psi), evidence(seen, dims))
    if den <= 1e-14:
        raise ValueError('zero probability evidence')
    return np.array([contract(psi, identity_mpo(psi),
                              evidence({**seen, missing: i}, dims)) / den
                     for i in range(dims[missing])])


def tt_scalar_count(cores):
    return sum(core.size for core in cores)


def tt_bond_ranks(cores):
    validate_cores(cores)
    return tuple(core.shape[-1] for core in cores[:-1])


def reconstruct(cores):
    """Debug/testing only: incurs exponential dense memory."""
    validate_cores(cores)
    out = cores[0][0]
    for core in cores[1:]:
        out = np.tensordot(out, core, axes=([-1], [0]))
    return out[..., 0]
