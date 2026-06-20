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

#include "triqs_xca/mixing.hpp"

#include <nda/linalg/inv.hpp>
#include <nda/linalg/matmul.hpp>
#include <nda/linalg/solve.hpp>
#include <nda/linalg/svd.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <utility>

namespace triqs_xca::mixing {

  namespace {

    bool is_finite(nda::dcomplex z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }

    bool is_finite(nda::array_const_view<nda::dcomplex, 1> a) {
      for (auto v : a) {
        if (!is_finite(v)) return false;
      }
      return true;
    }

    bool is_finite(nda::array<nda::dcomplex, 1> const &a) { return is_finite(nda::array_const_view<nda::dcomplex, 1>{a}); }

    void check_same_shape(nda::array_const_view<nda::dcomplex, 3> a, nda::array_const_view<nda::dcomplex, 3> b, char const *name_a,
                          char const *name_b) {
      if (a.shape() != b.shape()) throw std::runtime_error(std::string{name_a} + " and " + name_b + " must have the same shape");
      if (a.shape(1) != a.shape(2)) throw std::runtime_error(std::string{name_a} + " must be a rank-3 array of square matrices");
    }

    nda::matrix<nda::dcomplex> pseudo_inverse(nda::matrix<nda::dcomplex> const &a) {
      auto [u, s, vh] = nda::linalg::svd(a);
      auto n          = a.shape(1);
      auto m          = a.shape(0);
      auto sigma_p    = nda::matrix<nda::dcomplex>(n, m);
      sigma_p()       = 0.0;

      double smax = 0.0;
      for (auto sv : s) smax = std::max(smax, sv);
      double cutoff = std::numeric_limits<double>::epsilon() * static_cast<double>(std::max(m, n)) * smax;
      for (long i = 0; i < s.size(); ++i) {
        if (s(i) > cutoff) sigma_p(i, i) = 1.0 / s(i);
      }

      return nda::linalg::matmul(nda::linalg::matmul(transpose(conj(vh)), sigma_p), transpose(conj(u)));
    }

    nda::matrix<nda::dcomplex> inverse_or_pseudo_inverse(nda::matrix_const_view<nda::dcomplex> a) {
      try {
        return nda::linalg::inv(a);
      } catch (std::exception const &) { return pseudo_inverse(nda::matrix<nda::dcomplex>{a}); }
    }

  } // namespace

  diis_mixer::diis_mixer(int history_size, int start, double mix, std::optional<double> trust_radius, double regularization)
     : history_size_(history_size), start_(start), mix_(mix), trust_radius_(trust_radius), regularization_(regularization) {
    if (history_size_ < 1) throw std::runtime_error("history_size must be at least one");
    if (start_ < 1) throw std::runtime_error("start must be at least one");
    if (!(mix_ > 0.0 && mix_ <= 1.0)) throw std::runtime_error("mix must be in the interval (0, 1]");
    if (trust_radius_ && *trust_radius_ <= 0.0) throw std::runtime_error("trust_radius must be positive");
  }

  void diis_mixer::reset() {
    vectors_.clear();
    residuals_.clear();
    clear_last_result();
  }

  std::optional<nda::array<double, 1>> diis_mixer::last_coefficients() const {
    if (!has_last_coefficients_) return std::nullopt;
    return last_coefficients_;
  }

  nda::array<nda::dcomplex, 1> diis_mixer::linear_mix(nda::array_const_view<nda::dcomplex, 1> current,
                                                      nda::array_const_view<nda::dcomplex, 1> candidate) const {
    auto out = nda::array<nda::dcomplex, 1>(current.shape());
    out()    = (1.0 - mix_) * current + mix_ * candidate;
    return out;
  }

  void diis_mixer::clear_last_result() {
    has_last_coefficients_ = false;
    last_coefficients_     = nda::array<double, 1>{};
    last_used_diis_        = false;
  }

  nda::array<nda::dcomplex, 1> diis_mixer::update(nda::array_const_view<nda::dcomplex, 1> current,
                                                  nda::array_const_view<nda::dcomplex, 1> candidate) {
    if (current.shape() != candidate.shape()) throw std::runtime_error("current and candidate must have the same shape");
    auto residual = nda::array<nda::dcomplex, 1>(candidate.shape());
    residual()    = candidate - current;
    return update(current, candidate, residual);
  }

  nda::array<nda::dcomplex, 1> diis_mixer::update(nda::array_const_view<nda::dcomplex, 1> current, nda::array_const_view<nda::dcomplex, 1> candidate,
                                                  nda::array_const_view<nda::dcomplex, 1> residual) {
    if (current.shape() != candidate.shape()) throw std::runtime_error("current and candidate must have the same shape");
    if (residual.size() != candidate.size()) throw std::runtime_error("residual must have the same flattened size as candidate");

    if (!is_finite(residual)) {
      clear_last_result();
      return linear_mix(current, candidate);
    }

    residuals_.emplace_back(residual);
    vectors_.emplace_back(candidate);

    if (std::cmp_greater(residuals_.size(), history_size_)) {
      residuals_.erase(residuals_.begin());
      vectors_.erase(vectors_.begin());
    }

    if (std::cmp_less(residuals_.size(), start_)) {
      clear_last_result();
      return linear_mix(current, candidate);
    }

    auto mixed = diis_candidate();
    if (!mixed) {
      clear_last_result();
      return linear_mix(current, candidate);
    }

    last_used_diis_ = true;
    if (mix_ < 1.0) {
      auto out = nda::array<nda::dcomplex, 1>(candidate.shape());
      out()    = (1.0 - mix_) * current + mix_ * *mixed;
      return out;
    }

    return *mixed;
  }

  nda::array<nda::dcomplex, 1> diis_mixer::mix_from_coefficients(nda::array_const_view<double, 1> coeffs) const {
    auto mixed = nda::array<nda::dcomplex, 1>(vectors_.back().shape());
    mixed()    = 0.0;
    for (long i = 0; i < coeffs.size(); ++i) mixed() += coeffs(i) * vectors_[i];
    return mixed;
  }

  nda::array<double, 1> diis_mixer::restrict_step(nda::array_const_view<double, 1> coeffs) const {
    auto restricted = nda::array<double, 1>{coeffs};
    if (!trust_radius_) return restricted;

    restricted(restricted.size() - 1) -= 1.0;
    double step_norm = 0.0;
    for (double c : restricted) step_norm += c * c;
    step_norm = std::sqrt(step_norm);

    if (step_norm <= *trust_radius_) return nda::array<double, 1>{coeffs};

    restricted() *= *trust_radius_ / step_norm;
    restricted(restricted.size() - 1) += 1.0;
    return restricted;
  }

  std::optional<nda::array<nda::dcomplex, 1>> diis_mixer::diis_candidate() {
    auto n       = static_cast<long>(residuals_.size());
    auto gram    = nda::matrix<double>(n, n);
    double scale = 1.0;

    for (long i = 0; i < n; ++i) {
      for (long j = 0; j < n; ++j) {
        nda::dcomplex dot = 0.0;
        for (long k = 0; k < residuals_[i].size(); ++k) dot += conj(residuals_[i](k)) * residuals_[j](k);
        gram(i, j) = dot.real();
        scale      = std::max(scale, std::abs(gram(i, j)));
      }
    }

    gram() /= scale;
    for (long i = 0; i < n; ++i) gram(i, i) += regularization_;

    auto system                          = nda::matrix<double>(n + 1, n + 1);
    system()                             = 0.0;
    system(nda::range(n), nda::range(n)) = gram;
    for (long i = 0; i < n; ++i) {
      system(i, n) = 1.0;
      system(n, i) = 1.0;
    }

    auto rhs = nda::vector<double>(n + 1);
    rhs()    = 0.0;
    rhs(n)   = 1.0;

    nda::vector<double> sol;
    try {
      sol = nda::linalg::solve(system, rhs);
    } catch (std::exception const &) { return std::nullopt; }

    auto coeffs = nda::array<double, 1>(n);
    for (long i = 0; i < n; ++i) coeffs(i) = sol(i);
    coeffs = restrict_step(coeffs);

    auto mixed = mix_from_coefficients(coeffs);
    if (!is_finite(mixed)) return std::nullopt;

    last_coefficients_     = coeffs;
    has_last_coefficients_ = true;
    return mixed;
  }

  nda::array<nda::dcomplex, 3> cdiis_commutator_residual(double beta, cppdlr::imtime_ops const &itops, nda::array_const_view<nda::dcomplex, 3> G_iaa,
                                                         nda::array_const_view<nda::dcomplex, 3> G0_iaa,
                                                         nda::array_const_view<nda::dcomplex, 3> Sigma_iaa, double eta,
                                                         nda::matrix_const_view<nda::dcomplex> number_op, double dmu, bool symmetrize) {
    using cppdlr::_;

    check_same_shape(G_iaa, G0_iaa, "G_iaa", "G0_iaa");
    check_same_shape(G_iaa, Sigma_iaa, "G_iaa", "Sigma_iaa");
    if (number_op.shape(0) != G_iaa.shape(1) || number_op.shape(1) != G_iaa.shape(2))
      throw std::runtime_error("number_op must have shape compatible with G_iaa target matrices");

    auto r     = G_iaa.shape(0);
    auto n_orb = G_iaa.shape(1);

    auto ifops   = cppdlr::imfreq_ops(itops.lambda(), itops.get_rfnodes(), cppdlr::Fermion, symmetrize);
    auto G_w     = ifops.coefs2vals(beta, itops.vals2coefs(G_iaa));
    auto G0_w    = ifops.coefs2vals(beta, itops.vals2coefs(G0_iaa));
    auto Sigma_w = ifops.coefs2vals(beta, itops.vals2coefs(Sigma_iaa));

    auto cdiis_w      = nda::array<nda::dcomplex, 3>(r, n_orb, n_orb);
    auto identity     = nda::eye<nda::dcomplex>(n_orb);
    auto static_shift = nda::matrix<nda::dcomplex>(n_orb, n_orb);
    static_shift()    = eta * identity + dmu * number_op;

    for (long i = 0; i < r; ++i) {
      auto g0_inverse    = inverse_or_pseudo_inverse(G0_w(i, _, _));
      auto dyson_inverse = g0_inverse - static_shift - Sigma_w(i, _, _);
      cdiis_w(i, _, _)   = nda::linalg::matmul(G_w(i, _, _), dyson_inverse) - nda::linalg::matmul(dyson_inverse, G_w(i, _, _));
    }

    return itops.coefs2vals(ifops.vals2coefs(beta, cdiis_w));
  }

} // namespace triqs_xca::mixing
