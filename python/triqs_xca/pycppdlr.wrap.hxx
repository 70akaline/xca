#include <c2py/c2py.hpp>

#ifndef C2PY_HXX_DECLARATION_pycppdlr_GUARDS
#define C2PY_HXX_DECLARATION_pycppdlr_GUARDS
template <> constexpr bool c2py::is_wrapped<cppdlr::imtime_ops>     = true;
template <> inline constexpr auto c2py::tp_name<cppdlr::imtime_ops> = "triqs_xca.pycppdlr.ImTimeOps";
template <> constexpr bool c2py::is_wrapped<cppdlr::statistic_t>    = true;
template <>
const std::map<cppdlr::statistic_t, str_t> c2py::enum_to_string<cppdlr::statistic_t> = {{cppdlr::statistic_t::Boson, "Boson"},
                                                                                        {cppdlr::statistic_t::Fermion, "Fermion"}};
#endif