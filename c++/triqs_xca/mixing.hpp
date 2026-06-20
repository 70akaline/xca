/*******************************************************************************
 *
 * triqs_xca: Sum-Of-Exponentials bold HYBridization expansion impurity solver
 *
 * Copyright (C) 2026 by H. U.R. Strand
 *
 * triqs_xca is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * triqs_xca is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * triqs_xca. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#pragma once

#include <cppdlr/cppdlr.hpp>
#include <nda/nda.hpp>

#include <optional>
#include <vector>

namespace triqs_xca::mixing {

  /**
   * @brief Pulay DIIS mixer for fixed-point iterations.
   *
   * The mixer stores candidate vectors and residual vectors, then minimizes the
   * norm of the extrapolated residual under the usual sum(c) = 1 constraint.
   * Supplying the residual explicitly makes the same class usable for
   * commutator-DIIS (CDIIS).
   */
  class diis_mixer {
    public:
    diis_mixer(int history_size = 6, int start = 2, double mix = 1.0, std::optional<double> trust_radius = std::nullopt,
               double regularization = 1e-14);

    void reset();

    nda::array<nda::dcomplex, 1> update(nda::array_const_view<nda::dcomplex, 1> current, nda::array_const_view<nda::dcomplex, 1> candidate);

    nda::array<nda::dcomplex, 1> update(nda::array_const_view<nda::dcomplex, 1> current, nda::array_const_view<nda::dcomplex, 1> candidate,
                                        nda::array_const_view<nda::dcomplex, 1> residual);

    [[nodiscard]] int history_size() const noexcept { return history_size_; }
    [[nodiscard]] int start() const noexcept { return start_; }
    [[nodiscard]] double mix() const noexcept { return mix_; }
    [[nodiscard]] std::optional<double> trust_radius() const noexcept { return trust_radius_; }
    [[nodiscard]] double regularization() const noexcept { return regularization_; }
    [[nodiscard]] bool last_used_diis() const noexcept { return last_used_diis_; }
    [[nodiscard]] bool last_used_pulay() const noexcept { return last_used_diis_; }
    [[nodiscard]] std::optional<nda::array<double, 1>> last_coefficients() const;

    private:
    [[nodiscard]] nda::array<nda::dcomplex, 1> linear_mix(nda::array_const_view<nda::dcomplex, 1> current,
                                                          nda::array_const_view<nda::dcomplex, 1> candidate) const;
    std::optional<nda::array<nda::dcomplex, 1>> diis_candidate();
    [[nodiscard]] nda::array<nda::dcomplex, 1> mix_from_coefficients(nda::array_const_view<double, 1> coeffs) const;
    [[nodiscard]] nda::array<double, 1> restrict_step(nda::array_const_view<double, 1> coeffs) const;
    void clear_last_result();

    int history_size_;
    int start_;
    double mix_;
    std::optional<double> trust_radius_;
    double regularization_;

    std::vector<nda::array<nda::dcomplex, 1>> vectors_;
    std::vector<nda::array<nda::dcomplex, 1>> residuals_;
    nda::array<double, 1> last_coefficients_;
    bool has_last_coefficients_ = false;
    bool last_used_diis_        = false;
  };

  /**
   * @brief Green's-function commutator residual for CDIIS.
   *
   * Implements the residual C(iw) = [G(iw), G0^{-1}(iw) - eta I - dmu N -
   * Sigma(iw)] used by the iterative subspace algorithms of
   *
   * P. Pokhilko, C.-N. Yeh, and D. Zgid, J. Chem. Phys. 156, 094101 (2022),
   * DOI: 10.1063/5.0082586.
   *
   * The returned residual is transformed back to DLR imaginary-time nodes so it
   * can be used directly in the DIIS Gram matrix.
   */
  nda::array<nda::dcomplex, 3> cdiis_commutator_residual(double beta, cppdlr::imtime_ops const &itops, nda::array_const_view<nda::dcomplex, 3> G_iaa,
                                                         nda::array_const_view<nda::dcomplex, 3> G0_iaa,
                                                         nda::array_const_view<nda::dcomplex, 3> Sigma_iaa, double eta,
                                                         nda::matrix_const_view<nda::dcomplex> number_op, double dmu = 0.0, bool symmetrize = true);

} // namespace triqs_xca::mixing
