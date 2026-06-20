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

from triqs.operators import c, c_dag

from triqs_xca.solver import Solver


def solve_spinless(mixing):
    beta = 1.0
    lamb = 10.0
    eps = 1e-10
    tol = 1e-8
    mu = -0.01
    t = 1.0
    ek = 0.0

    H = -mu * c_dag(0, 0) * c(0, 0)
    S = Solver(beta, lamb, eps, H, [c(0, 0)], verbose=False)

    delta_iaa = t**2 * S.fd.free_greens(beta, np.array([[ek]]))
    S.set_hybridization(delta_iaa, verbose=False)

    kwargs = {}
    if mixing == 'cdiis':
        kwargs = dict(diis_history=3, diis_start=2, diis_trust_radius=1.0)

    diff = S.solve(
        1, tol=tol, maxiter=30, update_eta_exact=True,
        mixing=mixing, verbose=False, **kwargs)

    assert diff < tol
    assert np.isfinite(S.G_iaa).all()
    assert np.isfinite(S.Sigma_iaa).all()
    assert np.isfinite(S.eta)

    return S.G_iaa.copy(), S.Sigma_iaa.copy(), S.eta


def test_cdiis_spinless():
    G_linear, Sigma_linear, eta_linear = solve_spinless('linear')
    G_cdiis, Sigma_cdiis, eta_cdiis = solve_spinless('cdiis')

    np.testing.assert_allclose(G_cdiis, G_linear, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(Sigma_cdiis, Sigma_linear, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(eta_cdiis, eta_linear, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    test_cdiis_spinless()
