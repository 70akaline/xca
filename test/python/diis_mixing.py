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

import numpy as np

from triqs_xca.mixing import DIISMixer, PulayMixer


def test_diis_mixing():
    A = np.array([[0.45, 0.10], [-0.20, 0.35]])
    b = np.array([1.0, -0.5])
    x_ref = np.linalg.solve(np.eye(2) - A, b)

    def fixed_point(x):
        return A @ x + b

    mixer = DIISMixer(history_size=4, start=2)
    x = np.zeros(2)

    for _ in range(12):
        x_new = fixed_point(x)
        x = mixer.update(x, x_new)

    np.testing.assert_allclose(x, x_ref, atol=1e-11)
    assert mixer.last_used_diis


def test_diis_linear_startup():
    mixer = DIISMixer(history_size=4, start=3, mix=0.25)

    x = np.array([1.0, -1.0])
    x_new = np.array([3.0, 7.0])
    x_mixed = mixer.update(x, x_new)

    np.testing.assert_allclose(x_mixed, 0.75 * x + 0.25 * x_new)
    assert not mixer.last_used_diis


def test_diis_explicit_residual():
    mixer = DIISMixer(history_size=3, start=2)

    current = np.array([0.0, 0.0])
    first = np.array([1.0, 2.0])
    second = np.array([3.0, 4.0])

    mixed = mixer.update(current, first, residual=np.array([1.0, 0.0]))
    np.testing.assert_allclose(mixed, first)
    assert not mixer.last_used_diis

    mixed = mixer.update(first, second, residual=np.array([0.0, 1.0]))
    np.testing.assert_allclose(mixed, 0.5 * first + 0.5 * second, atol=1e-12)
    assert mixer.last_used_diis


def test_pulay_alias():
    assert PulayMixer is DIISMixer


def test_diis_trust_radius():
    mixer = DIISMixer(history_size=2, start=2, trust_radius=0.25)

    current = np.array([0.0])
    first = np.array([1.0])
    second = np.array([2.0])

    mixer.update(current, first, residual=np.array([1.0]))
    mixed = mixer.update(first, second, residual=np.array([0.99]))

    assert mixed[0] > second[0]
    assert mixed[0] < 3.0


if __name__ == '__main__':
    test_diis_mixing()
    test_diis_linear_startup()
    test_diis_explicit_residual()
    test_pulay_alias()
    test_diis_trust_radius()
