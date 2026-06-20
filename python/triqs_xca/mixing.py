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

from .pycppdlr import ImTimeOps as _ImTimeOps  # noqa: F401
from . import _mixing_cpp


class DIISMixer:
    """DIIS mixer for fixed-point iterations.

    The DIIS/Pulay history and constrained residual minimization live in the
    C++ ``triqs_xca::mixing::diis_mixer`` implementation. This Python class is
    a compatibility wrapper for the previous Python API.

    Given a current vector ``x`` and a candidate vector, this stores an error
    estimate and extrapolates from recent candidates. If no explicit residual
    is supplied, the error estimate is the fixed-point difference ``F(x) - x``.
    The coefficients minimize the norm of the extrapolated residual subject to
    ``sum(c) = 1``.

    The same C++ implementation is also used for commutator-DIIS (CDIIS) by
    supplying the commutator residual explicitly.
    """

    def __init__(self, history_size=6, start=2, mix=1.0, trust_radius=None,
                 regularization=1e-14):
        if history_size < 1:
            raise ValueError("history_size must be at least one")
        if start < 1:
            raise ValueError("start must be at least one")
        if not 0.0 < mix <= 1.0:
            raise ValueError("mix must be in the interval (0, 1]")
        if trust_radius is not None and trust_radius <= 0.0:
            raise ValueError("trust_radius must be positive")

        self.history_size = int(history_size)
        self.start = int(start)
        self.mix = float(mix)
        self.trust_radius = trust_radius
        self.regularization = float(regularization)

        self._mixer = _mixing_cpp.create_diis_mixer(
            history_size=self.history_size, start=self.start, mix=self.mix,
            trust_radius=self.trust_radius, regularization=self.regularization)

    def reset(self):
        _mixing_cpp.reset_diis_mixer(self._mixer)

    def update(self, current, candidate, residual=None):
        current = np.asarray(current)
        candidate = np.asarray(candidate)
        if current.shape != candidate.shape:
            raise ValueError("current and candidate must have the same shape")

        shape = candidate.shape
        current_vec = np.asarray(current, dtype=complex).reshape(-1)
        candidate_vec = np.asarray(candidate, dtype=complex).reshape(-1)

        if residual is None:
            residual_vec = None
        else:
            residual_vec = np.asarray(residual, dtype=complex).reshape(-1)
            if residual_vec.size != candidate_vec.size:
                raise ValueError("residual must have the same flattened size as candidate")

        mixed = _mixing_cpp.update_diis_mixer(
            self._mixer, current_vec, candidate_vec, residual=residual_vec)
        return np.asarray(mixed).reshape(shape)

    @property
    def last_coefficients(self):
        coeffs = _mixing_cpp.diis_last_coefficients(self._mixer)
        if coeffs is None:
            return None
        return np.asarray(coeffs)

    @property
    def last_used_diis(self):
        return bool(_mixing_cpp.diis_last_used(self._mixer))

    @property
    def last_used_pulay(self):
        return self.last_used_diis


PulayMixer = DIISMixer


def cdiis_commutator_residual(beta, itops, G_iaa, G0_iaa, Sigma_iaa, eta,
                              number_op, dmu=0.0, symmetrize=True):
    """Compute the Green's-function CDIIS commutator residual in C++.

    Implements the residual form used by P. Pokhilko, C.-N. Yeh, and D. Zgid,
    J. Chem. Phys. 156, 094101 (2022), DOI: 10.1063/5.0082586.
    """
    return _mixing_cpp.cdiis_commutator_residual(
        beta, itops, np.asarray(G_iaa, dtype=complex),
        np.asarray(G0_iaa, dtype=complex),
        np.asarray(Sigma_iaa, dtype=complex), float(eta),
        np.asarray(number_op, dtype=complex), dmu=float(dmu),
        symmetrize=bool(symmetrize))
