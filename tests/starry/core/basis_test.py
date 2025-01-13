import warnings

import numpy as np
import pytest

from jaxoplanet.starry.core.basis import A1, A2_inv, basis, poly_basis
from jaxoplanet.test_utils import assert_allclose

TOLERANCE = 1e-15


@pytest.mark.skip(reason="superseded by multi-precision tests")
@pytest.mark.parametrize("lmax", [10, 7, 5, 4, 3, 2, 1, 0])
def test_A1_symbolic(lmax):
    pytest.importorskip("sympy")
    expected = A1_symbolic(lmax) / (0.5 * np.sqrt(np.pi))
    calc = A1(lmax).todense()
    np.testing.assert_allclose(calc, expected, atol=5e-12)


@pytest.mark.skip(reason="superseded by multi-precision tests")
@pytest.mark.parametrize("lmax", [10, 7, 5, 4, 3, 2, 1, 0])
def test_A2_inv_symbolic(lmax):
    pytest.importorskip("sympy")
    expected = A2_inv_symbolic(lmax)
    calc = A2_inv(lmax).todense()
    np.testing.assert_allclose(calc, expected)


@pytest.mark.parametrize("lmax", [5, 4, 3, 2, 1, 0])
def test_A1(lmax):
    pytest.importorskip("mpmath")
    from jaxoplanet.starry.multiprecision import basis as mp_basis, utils

    expected = np.atleast_2d(utils.to_numpy(mp_basis.A1(lmax)))
    calc = np.atleast_2d(A1(lmax).todense())
    np.testing.assert_allclose(calc, expected)


@pytest.mark.parametrize("lmax", [15, 10, 7, 5, 4, 3, 2, 1, 0])
def test_A2_inv(lmax):
    pytest.importorskip("mpmath")
    from jaxoplanet.starry.multiprecision import basis as mp_basis, utils

    expected = np.atleast_2d(utils.to_numpy(mp_basis.A2_inv(lmax)))
    calc = np.atleast_2d(A2_inv(lmax).todense())
    np.testing.assert_allclose(calc, expected, atol=TOLERANCE)


@pytest.mark.parametrize("lmax", [15, 10, 7, 5, 4, 3, 2, 1, 0])
def test_direct_A2(lmax):
    pytest.importorskip("mpmath")
    from jaxoplanet.starry.multiprecision import basis as mp_basis, utils

    expected = np.atleast_2d(utils.to_numpy(mp_basis.A2(lmax)))
    calc = np.atleast_2d(np.linalg.inv(A2_inv(lmax).todense()))
    np.testing.assert_allclose(calc, expected, atol=TOLERANCE)


@pytest.mark.parametrize("lmax", [10, 7, 5, 4, 3, 2, 1, 0])
def test_A1_compare_starry(lmax):
    starry = pytest.importorskip("starry")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = starry.Map(lmax)
        expect = m.ops.A1.eval().toarray()
    calc = A1(lmax).todense()
    np.testing.assert_allclose(calc, expect, atol=5e-12)


@pytest.mark.parametrize("lmax", [10, 7, 5, 4, 3, 2, 1, 0])
def test_A2_inv_compare_starry(lmax):
    starry = pytest.importorskip("starry")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = starry.Map(lmax)
        A2 = m.ops.A.eval().toarray() @ m.ops.A1Inv.eval().toarray()
    inv = A2_inv(lmax)
    np.testing.assert_allclose(inv @ A2, np.eye(inv.shape[0]), atol=5e-12)


@pytest.mark.parametrize("lmax", [10, 7, 5, 4, 3, 2, 1, 0])
def test_basis_compare_starry(lmax):
    starry = pytest.importorskip("starry")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = starry.Map(lmax)
        expect = m.ops.A.eval().toarray()
    calc = basis(lmax).todense()
    assert_allclose(calc, expect)


def test_poly_basis():
    expect = np.array([1.0, 0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.06, 0.09])
    calc = poly_basis(2)(0.1, 0.3, 0.2)
    assert_allclose(calc, expect)


@pytest.mark.parametrize("lmax", [10, 5, 2, 1, 0])
def test_poly_basis_compare_starry(lmax):
    starry = pytest.importorskip("starry")

    x = np.linspace(-1, 1, 100)
    y = np.linspace(-0.5, 0.5, 100)
    z = np.linspace(-0.1, 0.1, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = starry.Map(lmax)
        expect = m.ops.pT(x, y, z)
    calc = poly_basis(lmax)(x, y, z)
    assert_allclose(calc, expect)


def A1_symbolic(lmax):
    """The sympy implementation of the A1 matrix from the starry paper"""
    import math

    import sympy as sm
    from sympy.functions.special.tensor_functions import KroneckerDelta

    x, y = sm.symbols("x y")

    def Coefficient(expression, term):
        """Return the coefficient multiplying `term` in `expression`."""
        coeff = expression.coeff(term)
        coeff = coeff.subs(sm.sqrt(1 - x**2 - y**2), 0).subs(x, 0).subs(y, 0)
        return coeff

    def ptilde(n, x, y):
        """Return the n^th term in the polynomial basis."""
        l = math.floor(math.sqrt(n))
        m = n - l * l - l
        mu = l - m
        nu = l + m
        if nu % 2 == 0:
            i = mu // 2
            j = nu // 2
            k = 0
        else:
            i = (mu - 1) // 2
            j = (nu - 1) // 2
            k = 1
        return x**i * y**j * sm.sqrt(1 - x**2 - y**2) ** k

    def A(l, m):
        """A spherical harmonic normalization constant."""
        return sm.sqrt(
            (2 - KroneckerDelta(m, 0))
            * (2 * l + 1)
            * sm.factorial(l - m)
            / (4 * sm.pi * sm.factorial(l + m))
        )

    def B(l, m, j, k):
        """Another spherical harmonic normalization constant."""
        try:
            ratio = sm.factorial((l + m + k - 1) / 2) / sm.factorial(
                (-l + m + k - 1) / 2
            )
        except ValueError:
            ratio = 0
        return (
            2**l
            * sm.factorial(m)
            / (
                sm.factorial(j)
                * sm.factorial(k)
                * sm.factorial(m - j)
                * sm.factorial(l - m - k)
            )
            * ratio
        )

    def C(p, q, k):
        """Return the binomial theorem coefficient `C`."""
        return sm.factorial(k / 2) / (
            sm.factorial(q / 2) * sm.factorial((k - p) / 2) * sm.factorial((p - q) / 2)
        )

    def Y(l, m, x, y):
        """Return the spherical harmonic of degree `l` and order `m`."""
        res = 0
        z = sm.sqrt(1 - x**2 - y**2)
        if m >= 0:
            for j in range(0, m + 1, 2):
                for k in range(0, l - m + 1, 2):
                    for p in range(0, k + 1, 2):
                        for q in range(0, p + 1, 2):
                            res += (
                                (-1) ** ((j + p) // 2)
                                * A(l, m)
                                * B(l, m, j, k)
                                * C(p, q, k)
                                * x ** (m - j + p - q)
                                * y ** (j + q)
                            )
                for k in range(1, l - m + 1, 2):
                    for p in range(0, k, 2):
                        for q in range(0, p + 1, 2):
                            res += (
                                (-1) ** ((j + p) // 2)
                                * A(l, m)
                                * B(l, m, j, k)
                                * C(p, q, k - 1)
                                * x ** (m - j + p - q)
                                * y ** (j + q)
                                * z
                            )
        else:
            for j in range(1, abs(m) + 1, 2):
                for k in range(0, l - abs(m) + 1, 2):
                    for p in range(0, k + 1, 2):
                        for q in range(0, p + 1, 2):
                            res += (
                                (-1) ** ((j + p - 1) // 2)
                                * A(l, abs(m))
                                * B(l, abs(m), j, k)
                                * C(p, q, k)
                                * x ** (abs(m) - j + p - q)
                                * y ** (j + q)
                            )
                for k in range(1, l - abs(m) + 1, 2):
                    for p in range(0, k, 2):
                        for q in range(0, p + 1, 2):
                            res += (
                                (-1) ** ((j + p - 1) // 2)
                                * A(l, abs(m))
                                * B(l, abs(m), j, k)
                                * C(p, q, k - 1)
                                * x ** (abs(m) - j + p - q)
                                * y ** (j + q)
                                * z
                            )

        return res

    def p_Y(l, m, lmax):
        ylm = Y(l, m, x, y)
        res = [ylm.subs(sm.sqrt(1 - x**2 - y**2), 0).subs(x, 0).subs(y, 0)]
        for n in range(1, (lmax + 1) ** 2):
            res.append(Coefficient(ylm, ptilde(n, x, y)))
        return res

    def A1(lmax):
        res = sm.zeros((lmax + 1) ** 2, (lmax + 1) ** 2)
        n = 0
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                res[n] = p_Y(l, m, lmax)
                n += 1
        return res

    return sm.lambdify([], A1(lmax))()


def A2_inv_symbolic(lmax):
    """The sympy implementation of the A2 matrix from the starry paper"""
    import math

    import sympy as sm

    x, y = sm.symbols("x y")

    def Coefficient(expression, term):
        """Return the coefficient multiplying `term` in `expression`."""
        coeff = expression.coeff(term)
        coeff = coeff.subs(sm.sqrt(1 - x**2 - y**2), 0).subs(x, 0).subs(y, 0)
        return coeff

    def ptilde(n, x, y):
        """Return the n^th term in the polynomial basis."""
        l = math.floor(math.sqrt(n))
        m = n - l * l - l
        mu = l - m
        nu = l + m
        if nu % 2 == 0:
            i = mu // 2
            j = nu // 2
            k = 0
        else:
            i = (mu - 1) // 2
            j = (nu - 1) // 2
            k = 1
        return x**i * y**j * sm.sqrt(1 - x**2 - y**2) ** k

    def gtilde(n, x, y):
        """Return the n^th term in the Green's basis."""
        l = math.floor(math.sqrt(n))
        m = n - l * l - l
        mu = l - m
        nu = l + m
        if nu % 2 == 0:
            I = [mu // 2]
            J = [nu // 2]
            K = [0]
            C = [(mu + 2) // 2]
        elif (l == 1) and (m == 0):
            I = [0]
            J = [0]
            K = [1]
            C = [1]
        elif (mu == 1) and (l % 2 == 0):
            I = [l - 2]
            J = [1]
            K = [1]
            C = [3]
        elif mu == 1:
            I = [l - 3, l - 1, l - 3]
            J = [0, 0, 2]
            K = [1, 1, 1]
            C = [-1, 1, 4]
        else:
            I = [(mu - 5) // 2, (mu - 5) // 2, (mu - 1) // 2]
            J = [(nu - 1) // 2, (nu + 3) // 2, (nu - 1) // 2]
            K = [1, 1, 1]
            C = [(mu - 3) // 2, -(mu - 3) // 2, -(mu + 3) // 2]

        res = 0
        for i, j, k, c in zip(I, J, K, C, strict=False):
            res += c * x**i * y**j * sm.sqrt(1 - x**2 - y**2) ** k
        return res

    def p_G(n, lmax):
        g = gtilde(n, x, y)
        res = [g.subs(sm.sqrt(1 - x**2 - y**2), 0).subs(x, 0).subs(y, 0)]
        for i in range(1, (lmax + 1) ** 2):
            res.append(Coefficient(g, ptilde(i, x, y)))
        return res

    def A2_inv(lmax):
        res = sm.zeros((lmax + 1) ** 2, (lmax + 1) ** 2)
        n = 0
        for l in range(lmax + 1):
            for _m in range(-l, l + 1):
                res[n] = p_G(n, lmax)
                n += 1
        return res

    return sm.lambdify([], A2_inv(lmax))()
