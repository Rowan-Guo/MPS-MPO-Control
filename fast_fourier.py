"""Contract a Fourier loss MPO with a categorical product-mixture probability TT.

The sine/cosine rotation blocks of each harmonic can be multiplied as complex
phases. Conditional on its latent mixture component, sites are independent,
so unobserved site phases can be averaged separately before multiplication.
This implements the same truncated Fourier operator as fourier_deadband_mpo.
"""
import numpy as np

from scalable import fourier_coefficients
from tensor import evidence


def affine_bounds(constants, contributions):
    lower = constants + contributions.min(axis=-1).sum(axis=-1)
    upper = constants + contributions.max(axis=-1).sum(axis=-1)
    return lower, upper


def fourier_domain(constants, contributions, deadband):
    lower, upper = affine_bounds(constants, contributions)
    return max(1.05 * float(np.max(np.maximum(-lower, upper))),
               1.01 * deadband, np.finfo(float).eps)


class FactorizedFourier:
    """Stored Fourier phases for (actions, monitors, sites, categories)."""

    def __init__(self, constants, contributions, deadband, harmonics, domain=None):
        constants = np.asarray(constants, dtype=float)
        contributions = np.asarray(contributions, dtype=float)
        if (constants.ndim != 2 or contributions.ndim != 4
                or contributions.shape[:2] != constants.shape
                or min(contributions.shape) < 1 or contributions.shape[-1] < 2
                or not np.isfinite(contributions).all() or not np.isfinite(constants).all()):
            raise ValueError('expected finite affine (actions, monitors, sites, categories)')
        if not isinstance(harmonics, (int, np.integer)) or harmonics < 1:
            raise ValueError('harmonics must be a positive integer')
        self.lower, self.upper = affine_bounds(constants, contributions)
        if domain is None:
            domain = fourier_domain(constants, contributions, deadband)
        if (not np.isfinite(domain) or np.any(self.lower < -domain - 1e-12)
                or np.any(self.upper > domain + 1e-12)):
            raise ValueError('Fourier domain must enclose all affine voltage deviations')
        self.domain = domain
        self.offset, coefficients = fourier_coefficients(deadband, domain, harmonics)
        self.frequencies = np.arange(1, harmonics + 1) * (np.pi / domain)
        self.coefficients = np.exp(1j * constants[..., None] * self.frequencies) * coefficients
        self.phases = np.exp(1j * contributions[..., None] * self.frequencies)

    @property
    def nbytes(self):
        """Resident numerical buffers, including coefficients and range checks."""
        return (self.phases.nbytes + self.coefficients.nbytes + self.frequencies.nbytes
                + self.lower.nbytes + self.upper.nbytes + np.asarray(self.offset).nbytes)

    def _conditional_moments(self, weights, local_probabilities, observed):
        local = np.asarray(local_probabilities, dtype=float)
        weights = np.asarray(weights, dtype=float)
        sites, categories = self.phases.shape[2:4]
        if (local.ndim != 3 or local.shape[1:] != (sites, categories)
                or weights.shape != (len(local),) or np.any(local < 0)
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1)
                or not np.allclose(local.sum(axis=2), 1)):
            raise ValueError('invalid categorical mixture')
        evidence(observed, (categories,) * sites)
        component_mass = weights.copy()
        moments = np.ones((len(weights), *self.coefficients.shape), dtype=complex)
        for site in range(sites):
            if site in observed:
                level = observed[site]
                component_mass *= local[:, site, level]
                moments *= self.phases[:, :, site, level]
            else:
                moments *= np.einsum('kd,amdh->kamh', local[:, site], self.phases[:, :, site])
        mass = component_mass.sum()
        if mass <= 1e-14:
            raise ValueError('zero probability evidence')
        posterior = component_mass / mass
        moments = np.einsum('k,kamh->amh', posterior, moments) * self.coefficients
        return moments, posterior, mass

    def scores(self, weights, local_probabilities, observed):
        """Conditional expected deadband loss for each action/monitor pair."""
        moments, _, _ = self._conditional_moments(weights, local_probabilities, observed)
        return self.offset + moments.real.sum(axis=-1)

    def scores_and_marginals(self, weights, local_probabilities, observed):
        """Also return the categorical posterior marginals for a plug-in baseline."""
        moments, posterior, mass = self._conditional_moments(weights, local_probabilities, observed)
        marginals = np.einsum('k,kid->id', posterior, local_probabilities)
        for site, value in observed.items():
            marginals[site] = 0.
            marginals[site, value] = 1.
        return self.offset + moments.real.sum(axis=-1), marginals, mass

    def shift_evaluator(self, weights, local_probabilities, observed):
        """Return a function evaluating deterministic voltage shifts for one evidence.

        The final two shift dimensions are (actions, monitors); an arbitrary
        number of leading batch dimensions is supported. The constructor's
        domain must cover the actual range of every shifted model.
        """
        moments, _, _ = self._conditional_moments(weights, local_probabilities, observed)

        def evaluate(shifts):
            shifts = np.asarray(shifts, dtype=float)
            if shifts.ndim < 2 or shifts.shape[-2:] != self.lower.shape:
                raise ValueError('shift array must end in (actions, monitors)')
            if (not np.isfinite(shifts).all() or np.any(self.lower + shifts < -self.domain - 1e-12)
                    or np.any(self.upper + shifts > self.domain + 1e-12)):
                raise ValueError('shifted voltage outside Fourier domain')
            phase = np.exp(1j * shifts[..., None] * self.frequencies)
            return self.offset + np.real(phase * moments).sum(axis=-1)

        return evaluate
