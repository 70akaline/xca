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

import copy

from triqs.gf import Gf, MeshDLRImFreq, inverse, iOmega_n, make_gf_dlr_imtime
from triqs.operators import n

from triqs_xca.block_sparse_solver import BlockSparseSolver


def make_symmetric_solver(beta=2.0, w_max=10.0, eps=1e-8):
    gf_struct = [['0', 1]]
    H_loc = 0 * n('0', 0)

    solver = BlockSparseSolver(
        H_loc, beta, w_max, eps, gf_struct=gf_struct,
        conserved_operators=[], dlr_symmetrize=True, verbose=False)

    mesh_w = MeshDLRImFreq(
        beta=beta, statistic='Fermion', w_max=w_max, eps=eps,
        symmetrize=True)
    delta_w = Gf(mesh=mesh_w, target_shape=[1, 1])
    delta_w << inverse(iOmega_n - 0.8)
    solver.Delta_tau['0'] << make_gf_dlr_imtime(delta_w)

    return solver


def test_symmetric_dlr_order_policy():
    solver = make_symmetric_solver()
    n_tau = len(solver.mesh_tau)
    solver.solve(max_order=2, maxiter=0, hyb_comp=False, verbose=False)
    assert solver.dlr_symmetrize
    assert len(solver.mesh_tau) == n_tau

    solver = make_symmetric_solver()
    n_tau = len(solver.mesh_tau)
    solver.solve(max_order=3, maxiter=0, hyb_comp=False, verbose=False)
    assert not solver.dlr_symmetrize
    assert solver.dlr_eps == 0.1 * solver.eps
    assert len(solver.mesh_tau) != n_tau

    solver_copy = copy.deepcopy(solver)
    assert not solver_copy.dlr_symmetrize
    assert solver_copy.dlr_eps == solver.dlr_eps
    assert len(solver_copy.mesh_tau) == len(solver.mesh_tau)
    solver_copy.solve_dyson(solver_copy.Sigma, solver_copy.eta)

    solver = make_symmetric_solver()
    n_tau = len(solver.mesh_tau)
    solver.solve(
        max_order=3, maxiter=0, hyb_comp=False, verbose=False,
        auto_convert_symmetric_dlr=False)
    assert solver.dlr_symmetrize
    assert len(solver.mesh_tau) == n_tau


if __name__ == "__main__":
    test_symmetric_dlr_order_policy()
