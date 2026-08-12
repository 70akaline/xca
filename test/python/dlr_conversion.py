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
import warnings
from unittest.mock import patch

import numpy as np

from triqs.gfs import (
    Gf, MeshDLR, MeshDLRImFreq, MeshDLRImTime, inverse, iOmega_n,
    make_gf_dlr, make_gf_dlr_imfreq, make_gf_dlr_imtime, make_gf_imfreq)
from triqs.operators import c, n

from triqs_xca.block_sparse_solver import BlockSparseSolver
from triqs_xca.dlr_conversion import copy_blockgf_to_mesh
from triqs_xca.dlr_conversion import resample_dlr_imtime_data_to_tau_rel
from triqs_xca.pycppdlr import ImTimeOps, build_dlr_rf
from triqs_xca.solver import Solver
from triqs_xca.triqs_solver import TriqsSolver


def make_delta_tau(beta, w_max, eps, symmetrize):
    mesh_w = MeshDLRImFreq(
        beta=beta, statistic='Fermion', w_max=w_max, eps=eps,
        symmetrize=symmetrize)
    delta_w = Gf(mesh=mesh_w, target_shape=[1, 1])
    delta_w << inverse(iOmega_n - 0.8)
    return make_gf_dlr_imtime(delta_w)


def make_symmetric_solver(beta=2.0, w_max=10.0, eps=1e-8):
    solver = BlockSparseSolver(
        0 * n('0', 0), beta, w_max, eps, gf_struct=[['0', 1]],
        conserved_operators=[], dlr_symmetrize=True, verbose=False)
    solver.Delta_tau['0'] << make_delta_tau(
        beta, w_max, eps, symmetrize=True)

    return solver


def make_dynamic_interaction_coefficients(
        beta, w_max, eps, symmetrize):
    mesh_tau = MeshDLRImTime(
        beta=beta, statistic='Boson', w_max=w_max, eps=eps,
        symmetrize=symmetrize)
    interaction_iw = make_gf_dlr_imfreq(
        Gf(mesh=mesh_tau, target_shape=[1, 1]))
    interaction_iw << -0.02 * inverse(1.0 - iOmega_n * iOmega_n)
    return make_gf_dlr(interaction_iw)


def assert_non_symmetric_internal_mesh(solver, mesh, eps):
    expected_mesh = MeshDLRImTime(
        beta=solver.beta, statistic='Fermion', w_max=solver.w_max,
        eps=0.1 * eps, symmetrize=False)
    np.testing.assert_allclose(
        np.array([float(t) for t in mesh]),
        np.array([float(t) for t in expected_mesh]))


def assert_converted_delta(solver, delta_tau, eps):
    expected = make_delta_tau(
        solver.beta, solver.w_max, 0.1 * eps, symmetrize=False)
    np.testing.assert_allclose(
        delta_tau['0'].data, expected.data, rtol=0, atol=10 * eps)


def assert_conversion_warning(caught):
    messages = [str(warning.message) for warning in caught]
    assert len(messages) == 1
    assert issubclass(caught[0].category, RuntimeWarning)
    assert 'violate causality' in messages[0]
    assert 'make the Dyson equation unstable or divergent' in messages[0]
    assert 'used only as the Dyson container' in messages[0]
    assert 'all internal calculations' in messages[0]
    assert 'non-symmetric DLR grid' in messages[0]


def test_symmetric_dlr_is_always_converted():
    for order in (1, 2, 3):
        solver = make_symmetric_solver()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            solver.solve(
                max_order=order, maxiter=0, hyb_comp=False, verbose=False)

        assert_conversion_warning(caught)
        assert not solver.dlr_symmetrize
        assert solver.dlr_eps == 0.1 * solver.eps
        assert_non_symmetric_internal_mesh(solver, solver.mesh_tau, solver.eps)
        assert_converted_delta(solver, solver.Delta_tau, solver.eps)

    solver_copy = copy.deepcopy(solver)
    assert not solver_copy.dlr_symmetrize
    assert solver_copy.dlr_eps == solver.dlr_eps
    assert len(solver_copy.mesh_tau) == len(solver.mesh_tau)
    G = solver_copy.solve_dyson(solver_copy.Sigma, solver_copy.eta)
    for _, block in G:
        assert np.isfinite(block.data).all()


def test_actual_block_sparse_solver_converts_symmetric_dlr():
    beta, w_max, eps = 2.0, 10.0, 1e-8
    solver = BlockSparseSolver(
        0 * n('0', 0), beta, w_max, eps, gf_struct=[['0', 1]],
        conserved_operators='automatic', dlr_symmetrize=True,
        verbose=False)
    solver.Delta_tau['0'] << make_delta_tau(
        beta, w_max, eps, symmetrize=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        solver.solve(
            max_order=1, maxiter=0, hyb_comp=False, verbose=False)

    assert_conversion_warning(caught)
    assert not solver.dlr_symmetrize
    assert_non_symmetric_internal_mesh(solver, solver.mesh_tau, solver.eps)


def test_symmetric_dlr_conversion_is_mpi_master_only():
    solver = make_symmetric_solver()
    with patch('triqs_xca.block_sparse_solver.is_root', return_value=False):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            solver.solve(
                max_order=1, maxiter=0, hyb_comp=False, verbose=False)

    assert not caught
    assert not solver.dlr_symmetrize


def test_symmetric_dlr_is_converted_for_bare_solver():
    solver = make_symmetric_solver()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        solver.solve_bare(max_order=1, hyb_comp=False, verbose=False)

    assert_conversion_warning(caught)
    assert not solver.dlr_symmetrize
    assert_non_symmetric_internal_mesh(solver, solver.mesh_tau, solver.eps)


def test_symmetric_dlr_is_converted_for_dyson_only():
    solver = make_symmetric_solver()
    symmetric_sigma = solver.Sigma.copy()
    for block_name, block in symmetric_sigma:
        block.data[:] = 0.05 * solver.G0[block_name].data

    reference = BlockSparseSolver(
        0 * n('0', 0), solver.beta, solver.w_max, 0.1 * solver.eps,
        gf_struct=[['0', 1]], conserved_operators=[],
        dlr_symmetrize=False, verbose=False)
    reference_sigma = reference.get_zero_pseudo_particle_propagator()
    copy_blockgf_to_mesh(
        symmetric_sigma, reference_sigma, solver.beta, solver.w_max,
        solver.eps, source_dlr_symmetrize=True)
    G_reference = reference.solve_dyson(reference_sigma, solver.eta)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        G = solver.solve_dyson(symmetric_sigma, solver.eta)

    assert_conversion_warning(caught)
    assert not solver.dlr_symmetrize
    assert_non_symmetric_internal_mesh(solver, solver.mesh_tau, solver.eps)
    for block_name, block in G:
        assert np.isfinite(block.data).all()
        np.testing.assert_allclose(
            block.data, G_reference[block_name].data, rtol=0,
            atol=10 * solver.eps)


def test_dynamic_interaction_coefficients_are_converted():
    solver = make_symmetric_solver()
    source = make_dynamic_interaction_coefficients(
        solver.beta, solver.w_max, solver.eps, symmetrize=True)
    solver.set_dynamic_interactions([n('0', 0)], source.data.copy())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        solver.fit_hybridization(compression=False, verbose=False)

    assert_conversion_warning(caught)
    assert solver.dynint_coeffs.shape[0] == len(solver.mesh_tau)

    target_mesh = MeshDLR(
        beta=solver.beta, statistic='Boson', w_max=solver.w_max,
        eps=solver.dlr_eps, symmetrize=False)
    converted = Gf(mesh=target_mesh, target_shape=[1, 1])
    converted.data[:] = solver.dynint_coeffs
    expected = make_dynamic_interaction_coefficients(
        solver.beta, solver.w_max, solver.dlr_eps, symmetrize=False)
    np.testing.assert_allclose(
        make_gf_imfreq(converted, 100).data,
        make_gf_imfreq(expected, 100).data,
        rtol=0, atol=10 * solver.eps)

    solver.init_diagram_evaluator()


def test_converted_dlr_eps_can_be_refined_after_fit():
    solver = make_symmetric_solver()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        solver.fit_hybridization(
            compression=False, verbose=False, converted_dlr_eps=1e-10)
    assert solver.dlr_eps == 1e-10

    solver.solve(
        max_order=1, maxiter=0, hyb_comp=False, verbose=False,
        converted_dlr_eps=1e-12)
    assert solver.dlr_eps == 1e-12


def test_symmetric_dlr_is_converted_for_triqs_solver():
    beta, w_max, eps = 2.0, 10.0, 1e-8
    for order in (1, 2, 3):
        solver = TriqsSolver(
            beta=beta, gf_struct=[('0', 1)], eps=eps, w_max=w_max,
            dlr_symmetrize=True, verbose=False)
        solver.Delta_tau['0'] << make_delta_tau(
            beta, w_max, eps, symmetrize=True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            solver.solve(
                h_int=0 * n('0', 0), order=order, maxiter=0,
                compress_hybridization=False, verbose=False)

        assert_conversion_warning(caught)
        assert not solver.dlr_symmetrize
        assert not solver.S.dlr_symmetrize
        assert solver.dlr_eps == 0.1 * solver.eps
        assert_non_symmetric_internal_mesh(solver, solver.tmesh, solver.eps)
        assert_converted_delta(solver, solver.Delta_tau, solver.eps)


def test_low_level_solver_never_uses_symmetric_dlr():
    beta, w_max, eps = 2.0, 10.0, 1e-8
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        solver = Solver(
            beta, beta * w_max, eps, 0 * n('0', 0), [c('0', 0)],
            dlr_symmetrize=True, verbose=False)

    assert_conversion_warning(caught)
    assert not solver.dlr_symmetrize
    assert solver.eps == 0.1 * eps
    G = solver.solve_dyson(
        np.zeros_like(solver.G_iaa), solver.eta, tol=10 * eps)
    assert np.isfinite(G).all()


def test_legacy_symmetric_low_level_solver_is_converted_on_restore():
    beta, w_max, eps = 2.0, 10.0, 1e-8
    reference = Solver(
        beta, beta * w_max, 0.1 * eps, 0 * n('0', 0), [c('0', 0)],
        verbose=False)

    source_rf = build_dlr_rf(beta * w_max, eps, symmetrize=True)
    source_ito = ImTimeOps(
        beta * w_max, source_rf, symmetrize=True)
    source_G = resample_dlr_imtime_data_to_tau_rel(
        reference.G_iaa, beta, w_max, source_ito.get_itnodes(),
        reference.eps, source_dlr_symmetrize=False)

    payload = reference.__reduce_to_dict__()
    payload.update({
        'eps': eps,
        'dlr_symmetrize': True,
        'dlr_rf': source_rf,
        'tau_i': beta * source_ito.get_itnodes(),
        'G0_iaa': source_G,
        'G_iaa': source_G.copy(),
    })

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        restored = Solver.__factory_from_dict__('Solver', payload)

    assert_conversion_warning(caught)
    assert not restored.dlr_symmetrize
    assert restored.eps == 0.1 * eps
    np.testing.assert_allclose(
        restored.G_iaa, reference.G_iaa, rtol=0, atol=10 * eps)
    G = restored.solve_dyson(
        np.zeros_like(restored.G_iaa), restored.eta, tol=10 * eps)
    assert np.isfinite(G).all()


if __name__ == "__main__":
    test_symmetric_dlr_is_always_converted()
    test_actual_block_sparse_solver_converts_symmetric_dlr()
    test_symmetric_dlr_conversion_is_mpi_master_only()
    test_symmetric_dlr_is_converted_for_bare_solver()
    test_symmetric_dlr_is_converted_for_dyson_only()
    test_dynamic_interaction_coefficients_are_converted()
    test_converted_dlr_eps_can_be_refined_after_fit()
    test_symmetric_dlr_is_converted_for_triqs_solver()
    test_low_level_solver_never_uses_symmetric_dlr()
    test_legacy_symmetric_low_level_solver_is_converted_on_restore()
