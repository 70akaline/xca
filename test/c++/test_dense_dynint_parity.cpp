#include <gtest/gtest.h>

#include <triqs/operators/many_body_operator.hpp>
#include <triqs/atom_diag/atom_diag.hpp>

#include <triqs_xca/atom_diag_utils.hpp>
#include <triqs_xca/dense_backbone.hpp>

using nda::dcomplex;

using cppdlr::_;
using cppdlr::build_dlr_rf;

using triqs::operators::many_body_operator_complex;
using triqs::operators::many_body_operator_real;
using triqs::operators::n;

using triqs_xca::atom_diag::ad_to_atom_prop;
using triqs_xca::dense::DenseDiagramEvaluator;

/**
 * All overloads of DenseDiagramEvaluator::compute_single_ptcle_gf() sum over the same set
 * of backbone flat indices, so they must give identical results in the presence of
 * dynamical interactions as well.
 *
 * This pins down the number of interaction operators, n_int, being handed to the
 * CorrelatorBackbone. With n_int = 0 every vertex is classified as fermionic in
 * Backbone::get_parity(), so backbones whose internal line is a dynamical interaction
 * (bosonic) line acquire the wrong permutation parity.
 *
 * The flat-index overloads used to omit n_int (dense_backbone.cpp:353, :373 and :387;
 * commit 63e072e fixed the omission at :358 and :413 but missed these). The error was
 * confined to the components of the returned array that involve the interaction
 * operators - the density-density correlator <T n(tau) n(0)> came out with a flipped sign
 * - which the Python solver never sees because it slices those components away. The
 * equivalent check at the Python level is test_dynint_chi in test/python/dynint.py.
 */

namespace {

  struct Model {
    triqs::atom_diag::atom_diag<true> ad;
    triqs::gfs::block_gf<triqs::mesh::dlr_imtime> G_ppsc;
    nda::vector<double> hyb_poles;
    nda::array<dcomplex, 3> hyb_coeffs;
    nda::array<dcomplex, 3> dynint_coeffs;
    std::vector<many_body_operator_real> dynint_ops;
  };

  /**
   * Single spinless level with a retarded interaction D(tau) coupled to the density.
   *
   * The dense evaluator requires a single atom_diag subspace (see the
   * assert(ad.n_subspaces() == 1) in dynint.cpp), which is what an empty list of conserved
   * operators gives us - the same setup the Python solver uses when
   * conserved_operators = [].
   */
  Model dynint_model(double beta, double Lambda, double eps) {

    many_body_operator_complex H = -0.3 * n("0", 0);

    triqs::atom_diag::fundamental_operator_set fop_set;
    fop_set.insert("0", 0);

    std::vector<many_body_operator_complex> sym_ops = {}; // no partitioning -> one subspace
    auto ad                                         = triqs::atom_diag::atom_diag<true>(H, fop_set, sym_ops);

    auto G_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);

    // Two-pole representation shared by the hybridization and the retarded interaction,
    // as produced by the joint adapol/DLR fit in the solver.
    int p = 2;
    nda::vector<double> hyb_poles(p);
    hyb_poles(0) = 1.3;
    hyb_poles(1) = -0.8;

    auto hyb_coeffs = nda::zeros<dcomplex>(p, 1, 1);
    hyb_coeffs(0, 0, 0) = 0.7;
    hyb_coeffs(1, 0, 0) = 0.5;

    auto dynint_coeffs = nda::zeros<dcomplex>(p, 1, 1);
    dynint_coeffs(0, 0, 0) = 0.9;
    dynint_coeffs(1, 0, 0) = 0.6;

    std::vector<many_body_operator_real> dynint_ops = {n<double>("0", 0)};

    return {.ad            = ad,
            .G_ppsc        = G_ppsc,
            .hyb_poles     = hyb_poles,
            .hyb_coeffs    = hyb_coeffs,
            .dynint_coeffs = dynint_coeffs,
            .dynint_ops    = dynint_ops};
  }

} // namespace

TEST(DenseDynint, spgf_flat_index_overloads_agree) {

  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-8;

  auto m = dynint_model(beta, Lambda, eps);

  nda::array<int, 2> topology = {{0, 2}, {1, 3}}; // second order: one internal line

  DenseDiagramEvaluator D(m.hyb_poles, m.hyb_coeffs, m.G_ppsc[0].mesh(), m.ad, m.dynint_ops, m.dynint_coeffs);

  ASSERT_EQ(D.n_hyb, 1);
  ASSERT_EQ(D.n_int, 1);
  ASSERT_EQ(D.n, 2);

  // Reference: sums all backbones internally, with n_int passed to the CorrelatorBackbone.
  auto spgf_all = D.compute_single_ptcle_gf(m.G_ppsc, topology);

  // Guard against a vacuous test: there must be backbones that put the interaction
  // operator on the internal line, i.e. bosonic vertices whose parity is at stake. (Note
  // that the total contribution of those backbones may well cancel - what this test
  // probes is the parity assigned to each of them individually.)
  DenseDiagramEvaluator D_no_dynint(m.hyb_poles, m.hyb_coeffs, m.G_ppsc[0].mesh(), m.ad);

  int n_backbones          = D.get_num_single_ptcle_gf_backbones(topology);
  int n_backbones_fermonic = D_no_dynint.get_num_single_ptcle_gf_backbones(topology);
  ASSERT_GT(n_backbones, n_backbones_fermonic) << "vacuous test: no backbone carries the interaction operator";

  // Path 1: the vector overload, used by the solver for the MPI distributed evaluation.
  nda::vector<int> f_ix_vec(n_backbones);
  for (int f_ix = 0; f_ix < n_backbones; ++f_ix) f_ix_vec(f_ix) = f_ix;
  auto spgf_vec = D.compute_single_ptcle_gf(m.G_ppsc, topology, f_ix_vec);

  // Path 2: accumulating the single flat index overload.
  auto spgf_single = nda::make_regular(0 * spgf_all);
  for (int f_ix = 0; f_ix < n_backbones; ++f_ix) spgf_single += D.compute_single_ptcle_gf(m.G_ppsc, topology, f_ix);

  // On failure, report which components disagree: the interaction components are the ones
  // at risk, since only they can put a bosonic vertex on an internal line.
  for (int mu = 0; mu < D.n; ++mu) {
    for (int kap = 0; kap < D.n; ++kap) {
      double err = nda::max_element(nda::abs(spgf_all(_, mu, kap) - spgf_vec(_, mu, kap)));
      if (err > 1.0e-12)
        std::cout << "component (" << mu << ", " << kap << "): max|all - vec| = " << err
                  << ", max|all| = " << nda::max_element(nda::abs(spgf_all(_, mu, kap))) << "\n";
    }
  }

  EXPECT_LE(nda::max_element(nda::abs(spgf_all - spgf_vec)), 1.0e-12)
     << "compute_single_ptcle_gf(G, topology, f_ix_vec) disagrees with compute_single_ptcle_gf(G, topology)";

  EXPECT_LE(nda::max_element(nda::abs(spgf_all - spgf_single)), 1.0e-12)
     << "compute_single_ptcle_gf(G, topology, f_ix) disagrees with compute_single_ptcle_gf(G, topology)";
}
