################################################################################
#
# triqs_xca: Sum-Of-Exponentials bold HYBridization expansion impurity solver
#
# Copyright (C) 2025 by H. U.R. Strand
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


import triqs.utility.mpi as mpi
from triqs.gfs import MeshDLR, MeshDLRImTime, BlockGf
from triqs.operators import c, Operator


from triqs_xca.solver import Solver, is_root
from triqs_xca.dlr_conversion import converted_dlr_eps as _converted_dlr_eps
from triqs_xca.dlr_conversion import copy_blockgf_to_mesh
from triqs_xca.dlr_conversion import resample_dlr_imtime_data
from triqs_xca.dlr_conversion import warn_symmetric_dlr_conversion


class TriqsSolver:

    r""" TRIQS Sum-Of-Exponentials bold HYBridization expansion impurity solver (triqs_xca)

    Parameters
    ----------

    beta : double
        inverse temperature

    gf_struct : list of pairs [ (str,int), ...]
        Structure of the Green's functions. It must be a
        list of pairs, each containing the name of the
        Green's function block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.

    eps : double
        Accuracy of the Discrete Lehmann Representation (DLR) imaginary time basis

    w_max : double
        Energy cut-off of the of the Discrete Lehmann Representation (DLR) imaginary time basis

    verbose : bool, optional
        Verbose printouts (default: `True`)

    """

    def __init__(self, beta, gf_struct, eps, w_max, dlr_symmetrize=False, verbose=True):

        self.verbose = verbose
        
        self.beta = beta
        self.gf_struct = gf_struct
        self.eps = eps
        self.w_max = w_max
        self.dlr_eps = eps
        self.dlr_symmetrize = dlr_symmetrize
        self._converted_from_symmetric_dlr = False

        self.dmesh = MeshDLR(beta=beta, statistic='Fermion', eps=self.dlr_eps, w_max=w_max, symmetrize=dlr_symmetrize)
        self.tmesh = MeshDLRImTime(beta=beta, statistic='Fermion', eps=self.dlr_eps, w_max=w_max, symmetrize=dlr_symmetrize)

        self.Delta_tau = BlockGf(mesh=self.tmesh, gf_struct=self.gf_struct)
        
        # gf_struct -> fundamental_operators
        
        fundamental_operators = []
        for s, n in gf_struct:
            fundamental_operators += [ c(s, i) for i in range(n) ]
            
        H_loc = 0 * Operator()
        
        lamb = beta * w_max
        internal_dlr_eps = (
            _converted_dlr_eps(eps) if dlr_symmetrize else eps)
        self.S = Solver(
            beta, lamb, internal_dlr_eps, H_loc, fundamental_operators,
            dlr_symmetrize=False, verbose=verbose)

        if not dlr_symmetrize:
            np.testing.assert_array_almost_equal(
                self.dmesh.values(), self.S.dlr_rf)
            np.testing.assert_array_almost_equal(
                self.tmesh.values(), self.S.tau_i)


    def solve(self, h_int, order, compress_hybridization=True,
              converted_dlr_eps=None, **kwargs):

        r""" Self-consistent solution of the pseudo-particle Green's function
        and pseudo-particle self-energy.

        Parameters
        ----------

        h_int : Triqs Operator
            Local many-body Hamiltonian of the impurity problem

        order : int
            Expansion order of the bold hybridization expansion

        compress_hybridization : bool, optional
            Use AAA compression of the hybridization function (default: `True`)
            (If `False` the DLR basis is used to represent the hybridization function.)
        converted_dlr_eps : float, optional
            DLR tolerance used when a symmetric input mesh is converted to the
            mandatory non-symmetric internal mesh. If not provided, it defaults
            to ``0.1 * eps``.

        tol : float, optional
            Pseudo-particle self-consistency convergence tolerance (default: `1e-9`)

        maxiter : int, optional
            Maximal number of self-consistent iterations (default: `100`)

        update_eta_exact : bool, optional
            Pseudo-particle energy shift update strategy (default: `True`)

        mix : float, optional
            Mixing ratio in the range [0, 1] (default: `1.0`)

        mixing : str, optional
            Outer pseudo-particle propagator mixing strategy, either `linear`
            `diis`, or `cdiis` (default: `linear`). `diis` uses the
            fixed-point difference residual. `cdiis` uses the Green's-function
            commutator residual in DLR Matsubara frequency, following the
            generalized CDIIS construction of Pokhilko et al.

        diis_history : int, optional
            Number of previous candidates used by DIIS/CDIIS mixing
            (default: `6`)

        diis_start : int, optional
            Number of residuals to collect before DIIS/CDIIS extrapolation starts
            (default: `2`)

        diis_trust_radius : float/None, optional
            Optional DIIS step restriction radius for the extrapolation
            coefficients, following the CDIIS step-restriction procedure.
            If `None`, no step restriction is applied (default: `None`)

        verbose : bool, optional
            Verbose printouts (default: `True`)

        G0_iaa : ndarray/None, optional
            Initial guess for the pseudo-particle propagator (default: `None`)

        """

        verbose = kwargs.get('verbose', True)
        kwargs['verbose'] = verbose

        if is_root() and verbose:

            print(f'max_order = {order}')

            def n_hybcomb(order):
                """ Number of hybridization function combinations at a given expansion order. """
                return self.S.fd.number_of_diagrams(order)

            def n_topologies(order):
                """ Number of diagram topologies for a given expansion order, """
                from .diag import all_connected_pairings
                num = 0
                for sign, diag in all_connected_pairings(order):
                    num += 1
                return num

            # -- Display number of topologies and hybridization combinations per order

            order_n_diags = [ (o, n_topologies(o), n_hybcomb(o)) for o in range(1,order+1) ]
            print(f'(Order, N_Topo, N_HybComb) = {order_n_diags}')

        self.order = order
        self.h_int = h_int

        self.__prepare_dlr_for_calculation(
            converted_dlr_eps=converted_dlr_eps)
        
        self.S.set_H_loc(h_int)
        self.S.G_iaa = self.S.G0_iaa.copy() # Fixme: use S.__setup_initial_guess?

        self.delta_iaa = self.__from_blockgf_to_array(self.Delta_tau)
        self.S.set_hybridization(self.delta_iaa, compress=compress_hybridization, verbose=verbose)
        
        self.S.solve(order, **kwargs)

        self.g_iaa = self.S.calc_spgf(order, verbose=verbose > 1)
        self.G_tau = self.__from_array_to_blockgf(self.g_iaa)

        if is_root() and hasattr(kwargs, 'verbose') and kwargs['verbose']:
            print(); self.S.timer.write()      

                
    def __from_blockgf_to_array(self, G):

        for b, g in G:
            assert( len(g.target_shape) == 2)
            assert( g.target_shape[0] == g.target_shape[1] )
            
        norb = sum([ g.target_shape[0] for b, g in G ])

        assert( norb == len(self.S.fundamental_operators) )
        
        ntau = len(G.mesh)
        g_iaa = np.zeros((ntau, norb, norb), dtype=complex)
        
        sidx = 0
        for b, g in G:
            size = g.target_shape[0]
            g_iaa[:, sidx:sidx+size, sidx:sidx+size] = g.data
            sidx += size

        return g_iaa

        
    def __from_array_to_blockgf(self, g_iaa):

        G = BlockGf(mesh=self.tmesh, gf_struct=self.gf_struct)
    
        sidx = 0
        for b, g in G:
            size = g.target_shape[0]
            g.data[:] = g_iaa[:, sidx:sidx+size, sidx:sidx+size]
            sidx += size

        return G


    def __auto_unsymmetrized_dlr_eps(self, converted_dlr_eps):
        return _converted_dlr_eps(self.eps, converted_dlr_eps)


    def __prepare_dlr_for_calculation(self, converted_dlr_eps=None):
        if not self.dlr_symmetrize:
            if (
                    self._converted_from_symmetric_dlr and
                    converted_dlr_eps is not None):
                dlr_eps = self.__auto_unsymmetrized_dlr_eps(
                    converted_dlr_eps)
                if dlr_eps != self.dlr_eps:
                    self.__convert_to_dlr_mesh(
                        dlr_symmetrize=False, dlr_eps=dlr_eps)
            return

        dlr_eps = self.__auto_unsymmetrized_dlr_eps(converted_dlr_eps)

        if is_root():
            warn_symmetric_dlr_conversion(self.dlr_eps, dlr_eps)

        self.__convert_to_dlr_mesh(dlr_symmetrize=False, dlr_eps=dlr_eps)
        self._converted_from_symmetric_dlr = True


    def __convert_to_dlr_mesh(self, dlr_symmetrize, dlr_eps):
        old_dlr_symmetrize = self.dlr_symmetrize
        old_dlr_eps = self.dlr_eps
        old_Delta_tau = self.Delta_tau

        self.dlr_eps = dlr_eps
        self.dlr_symmetrize = dlr_symmetrize
        self.dmesh = MeshDLR(
            beta=self.beta, statistic='Fermion', eps=self.dlr_eps,
            w_max=self.w_max, symmetrize=self.dlr_symmetrize)
        self.tmesh = MeshDLRImTime(
            beta=self.beta, statistic='Fermion', eps=self.dlr_eps,
            w_max=self.w_max, symmetrize=self.dlr_symmetrize)

        self.Delta_tau = BlockGf(mesh=self.tmesh, gf_struct=self.gf_struct)
        copy_blockgf_to_mesh(
            old_Delta_tau, self.Delta_tau, self.beta, self.w_max,
            old_dlr_eps, old_dlr_symmetrize)

        source_solver = self.S
        H_loc = getattr(self, 'h_int', source_solver.H_loc)
        G_iaa = None
        eta = None
        if hasattr(source_solver, 'G_iaa'):
            G_iaa = resample_dlr_imtime_data(
                source_solver.G_iaa, self.beta, self.w_max, self.tmesh,
                source_solver.eps, source_solver.dlr_symmetrize)
        if hasattr(source_solver, 'eta'):
            eta = source_solver.eta

        lamb = self.beta * self.w_max
        self.S = Solver(
            self.beta, lamb, self.dlr_eps, H_loc,
            source_solver.fundamental_operators, G_iaa=G_iaa, eta=eta,
            dlr_symmetrize=self.dlr_symmetrize, verbose=False)

        np.testing.assert_array_almost_equal(self.dmesh.values(), self.S.dlr_rf)
        np.testing.assert_array_almost_equal(self.tmesh.values(), self.S.tau_i)


    def __skip_keys(self):
        return []


    def __reduce_to_dict__(self):
        d = self.__dict__.copy()
        keys = set(d.keys()).intersection(self.__skip_keys())
        for key in keys: del d[key]
        return d


    @classmethod
    def __factory_from_dict__(cls, name, d):
        arg_keys = ['beta', 'gf_struct', 'eps', 'w_max']
        argv_keys = ['verbose']
        verbose = d['verbose']
        d['verbose'] = False # -- Suppress printouts on reconstruction from dict
        ret = cls(*[ d[key] for key in arg_keys ],
                  **{ key : d[key] for key in argv_keys })
        ret.__dict__.update(d)
        ret.verbose = verbose
        return ret


# -- Register Solver in Triqs formats

from h5.formats import register_class
register_class(TriqsSolver)
