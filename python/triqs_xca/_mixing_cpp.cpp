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

#include <c2py/c2py.hpp>
#include <cppdlr/cppdlr.hpp>
#include <nda/c2py/converters.hpp>

#include "pycppdlr.wrap.hxx"
#include "triqs_xca/mixing.hpp"

#include <array>
#include <memory>
#include <optional>
#include <string>

namespace {

  constexpr char const *diis_capsule_name = "triqs_xca.mixing.diis_mixer";

  using diis_mixer = triqs_xca::mixing::diis_mixer;

  diis_mixer *get_mixer(PyObject *capsule) {
    auto *ptr = static_cast<diis_mixer *>(PyCapsule_GetPointer(capsule, diis_capsule_name));
    if (ptr == nullptr && !PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "invalid DIIS mixer capsule");
    return ptr;
  }

  void diis_capsule_destructor(PyObject *capsule) {
    auto *ptr = static_cast<diis_mixer *>(PyCapsule_GetPointer(capsule, diis_capsule_name));
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    delete ptr;
  }

  void set_cpp_exception(std::exception const &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); }

  template <typename T> bool convertible(PyObject *obj) { return c2py::py_converter<T>::is_convertible(obj, true); }

  PyObject *create_diis_mixer(PyObject *, PyObject *args, PyObject *kwargs) {
    int history_size       = 6;
    int start              = 2;
    double mix             = 1.0;
    PyObject *trust_radius = Py_None;
    double regularization  = 1e-14;

    static constexpr std::array<char const *, 6> kwlist = {"history_size", "start", "mix", "trust_radius", "regularization", nullptr};

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iidOd", kwlist.data(), &history_size, &start, &mix, &trust_radius, &regularization))
      return nullptr;

    std::optional<double> trust;
    if (trust_radius != Py_None) {
      trust = PyFloat_AsDouble(trust_radius);
      if (PyErr_Occurred()) return nullptr;
    }

    try {
      auto ptr     = std::make_unique<diis_mixer>(history_size, start, mix, trust, regularization);
      auto capsule = PyCapsule_New(ptr.get(), diis_capsule_name, diis_capsule_destructor);
      if (capsule == nullptr) return nullptr;
      // NOLINTNEXTLINE(bugprone-unused-return-value, cppcoreguidelines-owning-memory)
      ptr.release();
      return capsule;
    } catch (std::exception const &e) {
      set_cpp_exception(e);
      return nullptr;
    }
  }

  PyObject *reset_diis_mixer(PyObject *, PyObject *args) {
    PyObject *capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return nullptr; // NOLINT(cppcoreguidelines-pro-type-vararg)
    auto *mixer = get_mixer(capsule);
    if (mixer == nullptr) return nullptr;
    mixer->reset();
    Py_RETURN_NONE;
  }

  PyObject *update_diis_mixer(PyObject *, PyObject *args, PyObject *kwargs) {
    PyObject *capsule   = nullptr;
    PyObject *current   = nullptr;
    PyObject *candidate = nullptr;
    PyObject *residual  = Py_None;

    static constexpr std::array<char const *, 5> kwlist = {"mixer", "current", "candidate", "residual", nullptr};

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|O", kwlist.data(), &capsule, &current, &candidate, &residual)) return nullptr;
    auto *mixer = get_mixer(capsule);
    if (mixer == nullptr) return nullptr;

    using vec_t = nda::array<nda::dcomplex, 1>;
    if (!convertible<vec_t>(current) || !convertible<vec_t>(candidate)) return nullptr;

    try {
      auto current_v   = c2py::py_converter<vec_t>::py2c(current);
      auto candidate_v = c2py::py_converter<vec_t>::py2c(candidate);
      vec_t mixed;
      if (residual == Py_None) {
        mixed = mixer->update(current_v, candidate_v);
      } else {
        if (!convertible<vec_t>(residual)) return nullptr;
        auto residual_v = c2py::py_converter<vec_t>::py2c(residual);
        mixed           = mixer->update(current_v, candidate_v, residual_v);
      }
      return c2py::py_converter<vec_t>::c2py(std::move(mixed));
    } catch (std::exception const &e) {
      set_cpp_exception(e);
      return nullptr;
    }
  }

  PyObject *diis_last_used(PyObject *, PyObject *args) {
    PyObject *capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return nullptr; // NOLINT(cppcoreguidelines-pro-type-vararg)
    auto *mixer = get_mixer(capsule);
    if (mixer == nullptr) return nullptr;
    if (mixer->last_used_diis())
      Py_RETURN_TRUE;
    else
      Py_RETURN_FALSE;
  }

  PyObject *diis_last_coefficients(PyObject *, PyObject *args) {
    PyObject *capsule = nullptr;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return nullptr; // NOLINT(cppcoreguidelines-pro-type-vararg)
    auto *mixer = get_mixer(capsule);
    if (mixer == nullptr) return nullptr;

    auto coeffs = mixer->last_coefficients();
    if (!coeffs) Py_RETURN_NONE;
    return c2py::py_converter<nda::array<double, 1>>::c2py(std::move(*coeffs));
  }

  PyObject *cdiis_commutator_residual(PyObject *, PyObject *args, PyObject *kwargs) {
    double beta         = 0.0;
    PyObject *itops_obj = nullptr;
    PyObject *G_obj     = nullptr;
    PyObject *G0_obj    = nullptr;
    PyObject *Sigma_obj = nullptr;
    double eta          = 0.0;
    PyObject *N_obj     = nullptr;
    double dmu          = 0.0;
    int symmetrize      = 1;

    static constexpr std::array<char const *, 10> kwlist = {"beta", "itops",     "G_iaa", "G0_iaa",     "Sigma_iaa",
                                                            "eta",  "number_op", "dmu",   "symmetrize", nullptr};

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dOOOOdO|dp", kwlist.data(), &beta, &itops_obj, &G_obj, &G0_obj, &Sigma_obj, &eta, &N_obj, &dmu,
                                     &symmetrize))
      return nullptr;

    using arr3_t = nda::array<nda::dcomplex, 3>;
    using mat_t  = nda::matrix<nda::dcomplex>;

    if (!convertible<cppdlr::imtime_ops>(itops_obj) || !convertible<arr3_t>(G_obj) || !convertible<arr3_t>(G0_obj) || !convertible<arr3_t>(Sigma_obj)
        || !convertible<mat_t>(N_obj))
      return nullptr;

    try {
      auto &itops = c2py::py_converter<cppdlr::imtime_ops>::py2c(itops_obj);
      auto G      = c2py::py_converter<arr3_t>::py2c(G_obj);
      auto G0     = c2py::py_converter<arr3_t>::py2c(G0_obj);
      auto Sigma  = c2py::py_converter<arr3_t>::py2c(Sigma_obj);
      auto N      = c2py::py_converter<mat_t>::py2c(N_obj);

      auto residual = triqs_xca::mixing::cdiis_commutator_residual(beta, itops, G, G0, Sigma, eta, N, dmu, symmetrize != 0);
      return c2py::py_converter<arr3_t>::c2py(std::move(residual));
    } catch (std::exception const &e) {
      set_cpp_exception(e);
      return nullptr;
    }
  }

  PyModuleDef *module_def() {
    static std::array<PyMethodDef, 7> module_methods = {
       {PyMethodDef{.ml_name  = "create_diis_mixer",
                    .ml_meth  = _PyCFunction_CAST(create_diis_mixer),
                    .ml_flags = METH_VARARGS | METH_KEYWORDS,
                    .ml_doc   = "Create a C++ DIIS mixer capsule."},
        PyMethodDef{.ml_name  = "reset_diis_mixer",
                    .ml_meth  = _PyCFunction_CAST(reset_diis_mixer),
                    .ml_flags = METH_VARARGS,
                    .ml_doc   = "Reset a C++ DIIS mixer."},
        PyMethodDef{.ml_name  = "update_diis_mixer",
                    .ml_meth  = _PyCFunction_CAST(update_diis_mixer),
                    .ml_flags = METH_VARARGS | METH_KEYWORDS,
                    .ml_doc   = "Update a vector with C++ DIIS/Pulay mixing."},
        PyMethodDef{.ml_name  = "diis_last_used",
                    .ml_meth  = _PyCFunction_CAST(diis_last_used),
                    .ml_flags = METH_VARARGS,
                    .ml_doc   = "Return whether the last update used DIIS."},
        PyMethodDef{.ml_name  = "diis_last_coefficients",
                    .ml_meth  = _PyCFunction_CAST(diis_last_coefficients),
                    .ml_flags = METH_VARARGS,
                    .ml_doc   = "Return the last DIIS coefficient vector, or None."},
        PyMethodDef{.ml_name  = "cdiis_commutator_residual",
                    .ml_meth  = _PyCFunction_CAST(cdiis_commutator_residual),
                    .ml_flags = METH_VARARGS | METH_KEYWORDS,
                    .ml_doc   = "Compute the Pokhilko-Yeh-Zgid Green's-function CDIIS residual in C++."},
        PyMethodDef{.ml_name = nullptr, .ml_meth = nullptr, .ml_flags = 0, .ml_doc = nullptr}}};

    static PyModuleDef def = {PyModuleDef_HEAD_INIT,
                              "_mixing_cpp",
                              "C++ DIIS/Pulay and CDIIS helpers for triqs_xca.",
                              -1,
                              module_methods.data(),
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr};
    return &def;
  }

} // namespace

// NOLINTNEXTLINE(bugprone-reserved-identifier)
extern "C" __attribute__((visibility("default"))) PyObject *PyInit__mixing_cpp() {
  if (!c2py::check_python_version("_mixing_cpp")) return nullptr;

#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  return PyModule_Create(module_def());
}
