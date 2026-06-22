import numpy as np


def converted_dlr_eps(eps, value=None):
    if value is not None:
        if value <= 0:
            raise ValueError('converted_dlr_eps must be positive.')
        return value
    return 0.1 * eps


def resample_dlr_imtime_data(
        data, beta, w_max, target_mesh,
        source_dlr_eps, source_dlr_symmetrize):
    from .pycppdlr import build_dlr_rf
    from .pycppdlr import ImTimeOps

    lamb = beta * w_max
    source_dlr_rf = build_dlr_rf(
        lamb, source_dlr_eps, source_dlr_symmetrize)
    source_ito = ImTimeOps(
        lamb, source_dlr_rf, symmetrize=source_dlr_symmetrize)
    coefs = source_ito.vals2coefs(np.asarray(data, dtype=complex))
    tau_rel = np.array([float(t) / beta for t in target_mesh])

    eval_at_tau = lambda t: source_ito.coefs2eval(coefs, t)
    return np.vectorize(eval_at_tau, signature='()->(m,m)')(tau_rel)


def copy_blockgf_to_mesh(
        source, target, beta, w_max,
        source_dlr_eps, source_dlr_symmetrize):
    for (_, src_g), (_, target_g) in zip(source, target):
        target_g.data[:] = resample_dlr_imtime_data(
            src_g.data, beta, w_max, target_g.mesh,
            source_dlr_eps, source_dlr_symmetrize)
