import warnings

import numpy as np


def warn_symmetric_dlr_conversion(source_eps, target_eps):
    warnings.warn(
        'Symmetric DLR input detected. A symmetric DLR representation may '
        'violate causality in subsequent Dyson iterations and make the Dyson '
        'equation unstable or divergent, even when it is used only as the '
        'Dyson container. triqs_xca currently converts the input to a '
        'non-symmetric DLR grid and performs all internal calculations on '
        f'that grid (eps {source_eps:2.2E} -> {target_eps:2.2E}).',
        RuntimeWarning,
        stacklevel=3)


def converted_dlr_eps(eps, value=None):
    if value is not None:
        if value <= 0:
            raise ValueError('converted_dlr_eps must be positive.')
        return value
    return 0.1 * eps


def resample_dlr_imtime_data(
        data, beta, w_max, target_mesh,
        source_dlr_eps, source_dlr_symmetrize):
    target_tau_rel = np.array([float(t) / beta for t in target_mesh])
    return resample_dlr_imtime_data_to_tau_rel(
        data, beta, w_max, target_tau_rel,
        source_dlr_eps, source_dlr_symmetrize)


def resample_dlr_imtime_data_to_tau_rel(
        data, beta, w_max, target_tau_rel,
        source_dlr_eps, source_dlr_symmetrize):
    from .pycppdlr import build_dlr_rf
    from .pycppdlr import ImTimeOps

    lamb = beta * w_max
    source_dlr_rf = build_dlr_rf(
        lamb, source_dlr_eps, source_dlr_symmetrize)
    source_ito = ImTimeOps(
        lamb, source_dlr_rf, symmetrize=source_dlr_symmetrize)
    coefs = source_ito.vals2coefs(np.asarray(data, dtype=complex))

    eval_at_tau = lambda t: source_ito.coefs2eval(coefs, t)
    return np.vectorize(eval_at_tau, signature='()->(m,m)')(
        np.asarray(target_tau_rel))


def resample_dlr_coefficients(
        data, beta, w_max, target_dlr_eps, target_dlr_symmetrize,
        source_dlr_eps, source_dlr_symmetrize, statistic):
    from triqs.gfs import (
        Gf, MeshDLR, make_gf_dlr, make_gf_dlr_imfreq, make_gf_imfreq)

    source_mesh = MeshDLR(
        beta=beta, statistic=statistic, w_max=w_max,
        eps=source_dlr_eps, symmetrize=source_dlr_symmetrize)
    if len(source_mesh) != np.asarray(data).shape[0]:
        raise ValueError(
            'DLR coefficient rank does not match the source DLR mesh.')

    source = Gf(mesh=source_mesh, target_shape=list(data.shape[1:]))
    source.data[:] = data

    n_iw = max(
        100,
        int(np.ceil(beta * w_max / (2 * np.pi))) + 10,
        4 * len(source_mesh))
    source_iw = make_gf_imfreq(source, n_iw)
    target_iw = make_gf_dlr_imfreq(
        source_iw, w_max, target_dlr_eps, target_dlr_symmetrize)
    return make_gf_dlr(target_iw).data.copy()


def copy_blockgf_to_mesh(
        source, target, beta, w_max,
        source_dlr_eps, source_dlr_symmetrize):
    for (_, src_g), (_, target_g) in zip(source, target):
        target_g.data[:] = resample_dlr_imtime_data(
            src_g.data, beta, w_max, target_g.mesh,
            source_dlr_eps, source_dlr_symmetrize)
