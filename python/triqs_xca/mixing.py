################################################################################
#
# triqs_xca: Sum-Of-Exponentials bold HYBridization expansion impurity solver
#
# Copyright (C) 2026 by H. U.R. Strand
#
# triqs_xca is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# triqs_xca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# triqs_xca. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

import numpy as np


class DIISMixer:
    """DIIS mixer for fixed-point iterations.

    Given a current vector ``x`` and a candidate vector, this stores an error
    estimate and extrapolates from recent candidates. If no explicit residual
    is supplied, the error estimate is the fixed-point difference ``F(x) - x``.
    The coefficients minimize the norm of the extrapolated residual subject to
    ``sum(c) = 1``.

    This is the standard Pulay DIIS construction. It can also be used for
    commutator-DIIS (CDIIS) by supplying the commutator residual explicitly.
    """

    def __init__(self, history_size=6, start=2, mix=1.0, trust_radius=None,
                 regularization=1e-14):
        if history_size < 1:
            raise ValueError("history_size must be at least one")
        if start < 1:
            raise ValueError("start must be at least one")
        if not 0.0 < mix <= 1.0:
            raise ValueError("mix must be in the interval (0, 1]")

        self.history_size = int(history_size)
        self.start = int(start)
        self.mix = float(mix)
        self.trust_radius = trust_radius
        self.regularization = float(regularization)

        self._vectors = []
        self._residuals = []
        self.last_coefficients = None
        self.last_used_diis = False

    def reset(self):
        self._vectors.clear()
        self._residuals.clear()
        self.last_coefficients = None
        self.last_used_diis = False

    @staticmethod
    def _linear_mix(current, candidate, mix):
        return (1.0 - mix) * current + mix * candidate

    def update(self, current, candidate, residual=None):
        current = np.asarray(current)
        candidate = np.asarray(candidate)
        if current.shape != candidate.shape:
            raise ValueError("current and candidate must have the same shape")

        if residual is None:
            residual = candidate - current

        residual = np.asarray(residual, dtype=complex).reshape(-1)
        vector = np.asarray(candidate, dtype=complex).reshape(-1)
        if not np.all(np.isfinite(residual)):
            self.last_coefficients = None
            self.last_used_diis = False
            return self._linear_mix(current, candidate, self.mix)

        self._residuals.append(residual.copy())
        self._vectors.append(vector.copy())
        if len(self._residuals) > self.history_size:
            self._residuals.pop(0)
            self._vectors.pop(0)

        if len(self._residuals) < self.start:
            self.last_coefficients = None
            self.last_used_diis = False
            return self._linear_mix(current, candidate, self.mix)

        diis = self._diis_candidate(candidate.shape)
        if diis is None:
            self.last_coefficients = None
            self.last_used_diis = False
            return self._linear_mix(current, candidate, self.mix)

        self.last_used_diis = True
        if self.mix < 1.0:
            return self._linear_mix(current, diis, self.mix)
        return diis

    @property
    def last_used_pulay(self):
        return self.last_used_diis

    def _diis_candidate(self, shape):
        n = len(self._residuals)
        gram = np.empty((n, n), dtype=float)

        for i, ri in enumerate(self._residuals):
            for j, rj in enumerate(self._residuals):
                gram[i, j] = np.vdot(ri, rj).real

        scale = max(float(np.max(np.abs(gram))), 1.0)
        gram /= scale
        gram.flat[::n + 1] += self.regularization

        system = np.zeros((n + 1, n + 1), dtype=float)
        system[:n, :n] = gram
        system[:n, n] = 1.0
        system[n, :n] = 1.0

        rhs = np.zeros(n + 1, dtype=float)
        rhs[n] = 1.0

        try:
            coeffs = np.linalg.solve(system, rhs)[:n]
        except np.linalg.LinAlgError:
            return None

        mixed = np.zeros_like(self._vectors[-1])
        for coeff, vector in zip(coeffs, self._vectors):
            mixed += coeff * vector

        if not np.all(np.isfinite(mixed)):
            return None

        if self.trust_radius is not None:
            mixed, coeffs = self._restrict_step(mixed, coeffs)

        self.last_coefficients = coeffs
        return mixed.reshape(shape)

    def _restrict_step(self, mixed, coeffs):
        radius = float(self.trust_radius)
        if radius <= 0.0:
            raise ValueError("trust_radius must be positive")

        step_coeffs = np.array(coeffs, copy=True)
        step_coeffs[-1] -= 1.0
        step_norm = np.linalg.norm(step_coeffs)
        if step_norm <= radius:
            return mixed, coeffs

        restricted = np.array(step_coeffs, copy=True)
        restricted *= radius / step_norm
        restricted[-1] += 1.0

        mixed = np.zeros_like(self._vectors[-1])
        for coeff, vector in zip(restricted, self._vectors):
            mixed += coeff * vector

        return mixed, restricted


PulayMixer = DIISMixer
