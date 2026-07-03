
import numpy as np


from triqs.utility import mpi
from triqs.gfs import MeshDLRImTime


class Dummy:
    def __init__(self): pass


def run_calc(m_dlr, ed_solver, xca_solver, max_order=1, t=1.0):
    
    """ Run a single calculation of the dynamic interaction expansion 
    and compare to ED reference solution for a single fermionic level 
    coupled to another fermionic level. """

    ed = ed_solver(m_dlr, t=t)

    ed = ed_solver(m_dlr, t=t)

    oxs = []
    for order in range(1, max_order+1):
         print(f'Computing XCA solution for order {order}...')
         ox = xca_solver(m_dlr, t=t, sigma_order=order, verbose=True)
         oxs.append(ox)

    d = Dummy()
    d.ed = ed
    d.oxs = oxs
    return d


def plot_comparison(m_dlr, ed_solver, xca_solver, max_order=4, t=-0.5):

    """ Compare XCA solution to ED reference solution for a single fermionic level 
    coupled to another fermionic level. """

    d = run_calc(m_dlr, ed_solver, xca_solver, max_order=max_order, t=t)
    ed, oxs = d.ed, d.oxs

    if mpi.is_master_node():

        for ox in oxs:
            G_err = np.max(np.abs((ox.G_tau - ed.G_tau).data))
            Chi_err = np.max(np.abs((ox.Chi_tau - ed.Chi_tau).data))
            print(f'O{ox.sigma_order} error: G={G_err:.3e}, Chi={Chi_err:.3e}')

    if mpi.is_master_node():

        from triqs.plot.mpl_interface import oplot, plt
        plt.figure(figsize=(6, 8))
        subp = [4, 1, 1]

        plt.subplot(*subp); subp[-1] += 1
        for ox in oxs:
            oplot(ox.G_tau.real, marker='+', label=f'O{ox.sigma_order} sc')
        oplot(ed.G_tau.real, marker='x', lw=4., alpha=0.5, label='ED')
        plt.ylabel(r'$g(\tau)$')
        plt.ylim(top=0.)

        plt.subplot(*subp); subp[-1] += 1
        for ox in oxs:
            oplot(ox.G_tau.real - ed.G_tau.real, marker='+', label=f'O{ox.sigma_order} sc')
        plt.ylabel(r'Err $g(\tau)$')
        plt.ylim(top=0.)

        plt.subplot(*subp); subp[-1] += 1
        for ox in oxs:
            oplot(ox.Chi_tau.real, marker='+', label=f'O{ox.spgf_order} sc')
        oplot(ed.Chi_tau.real, marker='x', lw=4., alpha=0.5, label='ED')
        plt.ylabel(r'$\chi_{nn}(\tau)$')

        plt.subplot(*subp); subp[-1] += 1
        for ox in oxs:
            oplot(ox.Chi_tau.real - ed.Chi_tau.real, marker='+', label=f'O{ox.spgf_order} sc')
        plt.ylabel(r'Err $\chi_{nn}(\tau)$')

        plt.tight_layout()
        plt.show()


def run_convergence_calc(m_dlr, ed_solver, xca_solver, max_order=4, t2s=np.logspace(-1.5, 0.5, 4)):

    """ Run a convergence calculation of the dynamic interaction expansion 
    and compare to ED reference solution for a single fermionic level 
    coupled to another fermionic level. """

    G_errs_t = []
    Chi_errs_t = []

    for i, t2 in enumerate(t2s):

        t = -np.sqrt(t2)
        d = run_calc(m_dlr, ed_solver, xca_solver, max_order=max_order, t=t)
        ed, oxs = d.ed, d.oxs

        G_errs = np.zeros(max_order)
        Chi_errs = np.zeros(max_order)

        for ox in oxs:
            G_errs[ox.sigma_order-1] = np.max(np.abs((ox.G_tau - ed.G_tau).data))
            Chi_errs[ox.sigma_order-1] = np.max(np.abs((ox.Chi_tau - ed.Chi_tau).data))

        G_errs_t.append(G_errs)
        Chi_errs_t.append(Chi_errs)

    orders = list(range(1, max_order + 1))

    # Transpose G and Chi
    G_errss = [ np.array([G_errs_t[j][i] for j in range(len(t2s))]) for i in range(max_order) ]
    Chi_errss = [ np.array([Chi_errs_t[j][i] for j in range(len(t2s))]) for i in range(max_order) ]

    return orders, G_errss, Chi_errss


def test_convergence_rate(m_dlr, ed_solver, xca_solver, label='dimer', max_order=5, do_test=False, verbose=True):

    """ Test convergence rate of the dynamic interaction expansion 
    by comparing to ED reference solution for a single fermionic level 
    coupled to another fermionic level. """

    t2s = np.logspace(-1.5, 0.5, 4)
    orders, G_errss, Chi_errss = run_convergence_calc(m_dlr, ed_solver, xca_solver, max_order=max_order, t2s=t2s)

    if mpi.is_master_node():
        # Compute convergence rates
        G_rates = []
        Chi_rates = []
        for order, G_errs, Chi_errs in zip(orders, G_errss, Chi_errss):
            G_rate = (np.log(G_errs[:-1] / G_errs[1:]) / np.log(t2s[:-1] / t2s[1:]))[0]
            Chi_rate = (np.log(Chi_errs[:-1] / Chi_errs[1:]) / np.log(t2s[:-1] / t2s[1:]))[0]
            G_rates.append(G_rate)
            Chi_rates.append(Chi_rate)
            print(f'Order {order} convergence rates: G={G_rate:.2f}, Chi={Chi_rate:.2f}')

    if verbose and mpi.is_master_node():
        # Plot errors
        import matplotlib.pyplot as plt

        for i, (order, G_errs, Chi_errs) in enumerate(zip(orders, G_errss, Chi_errss)):

            x = [t2s[0], t2s[0] + 0.2 * (t2s[-1] - t2s[0])]
            y = (x / x[0])**order

            subp = [1, 2, 1]
            plt.subplot(*subp); subp[-1] += 1
            plt.loglog(t2s, G_errs, 'o-', label=f'O{order}', alpha=0.75)
            plt.plot(x, G_errs[0] * y , 'k--', lw=0.5)
            plt.xlabel('$t^2$')
            plt.ylabel(r'Error: $\max_i |g(\tau_i) - g^{\text{ED}}(\tau_i)|$')
            plt.legend(loc='best')
            plt.grid(True)
            plt.axis('equal')

            plt.subplot(*subp); subp[-1] += 1
            plt.loglog(t2s, Chi_errs, 'o-', label=f'O{order}', alpha=0.75)
            plt.plot(x, Chi_errs[0] * y, 'k--', lw=0.5)
            plt.xlabel('$t^2$')
            plt.ylabel(r'Error: $\max_i |\chi_{nn}(\tau_i) - \chi_{nn}^{\text{ED}}(\tau_i)|$')
            plt.legend(loc='best')
            plt.grid(True)
            plt.axis('equal')

        plt.tight_layout()
        plt.savefig(f'figure_{label}_convergence.pdf')
        plt.show()

    if mpi.is_master_node():
        # Check convergence rates
        for order, G_rate, Chi_rate in zip(orders, G_rates, Chi_rates):
            print(f'Order {order} convergence rates: G={G_rate:.2f}, Chi={Chi_rate:.2f}')
            if do_test:
                assert( np.abs(G_rate - order) < 0.3 ), f'Expected convergence rate of {order} for order {order}, but got {G_rate}, diff {np.abs(G_rate - order)}'
                assert( np.abs(Chi_rate - order) < 0.3 ), f'Expected convergence rate of {order} for order {order}, but got {Chi_rate}, diff {np.abs(Chi_rate - order)}'
