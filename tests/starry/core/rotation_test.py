import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoplanet.starry.core.rotation import (
    dot_rotation_matrix,
    full_rotation_axis_angle,
    left_project,
    right_project,
)
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.skip(reason="superseded by multi-precision tests")
@pytest.mark.parametrize("l_max", [5, 4, 3, 2, 1, 0])
@pytest.mark.parametrize("u", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)])
@pytest.mark.parametrize("theta", [0.1])
def test_dot_rotation_symbolic(l_max, u, theta):
    """Test full rotation matrix against symbolic one"""
    pytest.importorskip("sympy")
    ident = np.eye(l_max**2 + 2 * l_max + 1)
    expected = np.array(R_symbolic(l_max, u, theta)).astype(float)
    calc = dot_rotation_matrix(l_max, u[0], u[1], u[2], theta)(ident)
    assert_allclose(calc, expected)


@pytest.mark.parametrize("l_max", [5, 4, 3, 2, 1, 0])
@pytest.mark.parametrize("theta", [-0.5, 0.0, 0.1, 1.5 * np.pi])
def test_dot_rotation_z(l_max, theta):
    ident = np.eye(l_max**2 + 2 * l_max + 1)
    expected = dot_rotation_matrix(l_max, 0.0, 0.0, 1.0, theta)(ident)
    calc = dot_rotation_matrix(l_max, None, None, 1.0, theta)(ident)
    assert_allclose(calc, expected)


@pytest.mark.parametrize("l_max", [5, 4, 3, 2, 1, 0])
@pytest.mark.parametrize("u", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)])
@pytest.mark.parametrize("theta", [0.1])
def test_R(l_max, u, theta):
    """Test full rotation matrix against symbolic one"""
    pytest.importorskip("mpmath")
    from jaxoplanet.starry.core.rotation import compute_rotation_matrices
    from jaxoplanet.starry.multiprecision import utils
    from jaxoplanet.starry.multiprecision.rotation import R

    expected = R(l_max, u, theta)
    calc = compute_rotation_matrices(l_max, u[0], u[1], u[2], theta)
    for e, c in zip(expected, calc, strict=False):
        assert_allclose(c, np.atleast_2d(utils.to_numpy(e)))


def test_dot_rotation_negative():
    starry = pytest.importorskip("starry")
    l_max = 5
    n_max = l_max**2 + 2 * l_max + 1
    y = np.linspace(-1, 1, n_max)
    starry_op = starry._core.core.OpsYlm(l_max, 0, 0, 1)
    expected = starry_op.dotR(y[None, :], 1.0, 0, 0.0, -0.5 * np.pi)[0]
    calc = dot_rotation_matrix(l_max, 1.0, 0.0, 0.0, -0.5 * np.pi)(y)
    assert_allclose(calc, expected)


def test_dot_rotation_edge_cases():
    l_max = 5
    n_max = l_max**2 + 2 * l_max + 1
    ident = np.eye(n_max)
    assert np.all(np.isfinite(dot_rotation_matrix(l_max, 0, 0, 1, 0)(ident)))

    g = jax.grad(
        lambda *args: dot_rotation_matrix(l_max, *args)(ident).sum(),
        argnums=(0, 1, 2, 3),
    )(0.0, 0.0, 1.0, 0.0)
    for g_ in g:
        assert np.all(np.isfinite(g_))


def test_right_project_axis_angle_edge_case():
    theta = theta_z = obl = 0.0
    inc = jnp.pi / 2

    x = jnp.array(full_rotation_axis_angle(inc, obl, theta, theta_z))
    assert jnp.all(jnp.isfinite(x))


@pytest.mark.parametrize("l_max", [10, 7, 5, 4])
@pytest.mark.parametrize("u", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)])
@pytest.mark.parametrize("theta", [0.1])
def test_dot_rotation_compare_starry(l_max, u, theta):
    """Comparison test with starry OpsYlm.dotR"""
    starry = pytest.importorskip("starry")
    random = np.random.default_rng(l_max)
    n_max = l_max**2 + 2 * l_max + 1
    M1 = np.eye(n_max)
    M2 = random.normal(size=(5, n_max))
    starry_op = starry._core.core.OpsYlm(l_max, 0, 0, 1)
    jax_op = dot_rotation_matrix(l_max, *u, theta)
    for M in [M1, M2]:
        norm = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        expected = starry_op.dotR(M, u[0] / norm, u[1] / norm, u[2] / norm, theta)
        calc = jax_op(M)
        assert_allclose(calc, expected)


def right_project_reference(deg, inc, obl, theta, theta_z, x):
    x = dot_rotation_matrix(deg, 0, 0, 1.0, theta_z)(x)
    x = dot_rotation_matrix(
        deg, -jnp.cos(obl), -jnp.sin(obl), 0.0, -(0.5 * jnp.pi - inc)
    )(x)
    x = dot_rotation_matrix(deg, None, None, 1.0, obl)(x)
    x = dot_rotation_matrix(deg, 1.0, 0.0, 0.0, -0.5 * jnp.pi)(x)
    x = dot_rotation_matrix(deg, None, None, 1.0, theta)(x)
    x = dot_rotation_matrix(deg, 1.0, 0.0, 0.0, 0.5 * jnp.pi)(x)
    return x


def left_project_reference(deg, inc, obl, theta, theta_z, x):
    x = dot_rotation_matrix(deg, 1.0, 0.0, 0.0, -0.5 * jnp.pi)(x)
    x = dot_rotation_matrix(deg, None, None, 1.0, -theta)(x)
    x = dot_rotation_matrix(deg, 1.0, 0.0, 0.0, 0.5 * jnp.pi)(x)
    x = dot_rotation_matrix(deg, None, None, 1.0, -obl)(x)
    x = dot_rotation_matrix(
        deg, -jnp.cos(obl), -jnp.sin(obl), 0.0, (0.5 * jnp.pi - inc)
    )(x)
    x = dot_rotation_matrix(deg, 0, 0, 1.0, -theta_z)(x)
    return x


@pytest.mark.parametrize("deg", [5, 2, 1, 0])
@pytest.mark.parametrize(
    "angles",
    [
        (0.1, 0.2, 0.3, 0.1),
        (0.1, -0.2, 0.3, 0.0),
        (0, 0, 0, 0.1),
        (-0.1, 0, 0, 0.0),
        (0, 0.4, 0, 0.1),
        (0, 0, 0.5, 0.0),
        (0.3, 0.3, None, 0.0),
        (0.3, 0.3, 0.5, None),
        (0.3, 0.3, None, None),
        (0.1, None, 0.1, 0.1),
        (None, None, 0.1, 0.1),
        (None, None, None, None),
    ],
)
def test_right_project(deg, angles):
    n_max = deg**2 + 2 * deg + 1
    _angles = [a if a is not None else 0.0 for a in angles]
    expect = right_project_reference(deg, *_angles, jnp.ones(n_max))
    calc = right_project(deg, *angles, jnp.ones(n_max))
    assert_allclose(calc, expect)


@pytest.mark.parametrize("deg", [5, 2, 1, 0])
@pytest.mark.parametrize(
    "angles",
    [
        (0.1, 0.2, 0.3, 0.1),
        (0.1, -0.2, 0.3, 0.0),
        (0, 0, 0, 0.1),
        (-0.1, 0, 0, 0.0),
        (0, 0.4, 0, 0.1),
        (0, 0, 0.5, 0.0),
        (0.3, 0.3, None, 0.0),
        (0.3, 0.3, 0.5, None),
        (0.3, 0.3, None, None),
        (0.1, None, 0.1, 0.1),
        (None, None, 0.1, 0.1),
        (None, None, None, None),
    ],
)
def test_left_project(deg, angles):
    n_max = deg**2 + 2 * deg + 1
    _angles = [a if a is not None else 0.0 for a in angles]
    expect = left_project_reference(deg, *_angles, jnp.ones(n_max))
    calc = left_project(deg, *angles, jnp.ones(n_max))
    assert_allclose(calc, expect)


@pytest.mark.skip(reason="superseded by multi-precision tests")
@pytest.mark.parametrize("l_max", [5, 4, 3, 2, 1, 0])
@pytest.mark.parametrize("u", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)])
@pytest.mark.parametrize("theta", [0.1])
def test_R_multiprecision_symbolic(l_max, u, theta):
    pytest.importorskip("sympy")
    from scipy.linalg import block_diag

    from jaxoplanet.starry.multiprecision import utils
    from jaxoplanet.starry.multiprecision.rotation import R

    expected = np.array(R_symbolic(l_max, u, theta)).astype(float)
    calc = block_diag(
        *[np.atleast_2d(utils.to_numpy(r)).astype(float) for r in R(l_max, u, theta)]
    )
    assert_allclose(calc, expected)


def R_symbolic(lmax, u, theta):
    import sympy as sm
    from sympy.functions.special.tensor_functions import KroneckerDelta

    def Dmn(l, m, n, alpha, beta, gamma):
        """Compute the (m, n) term of the Wigner D matrix."""
        sumterm = 0
        # Expression diverges when beta = 0
        if beta == 0:
            beta = 1e-16
        for k in range(l + m + 1):
            sumterm += (
                (-1) ** k
                * sm.cos(beta / 2) ** (2 * l + m - n - 2 * k)
                * sm.sin(beta / 2) ** (-m + n + 2 * k)
                / (
                    sm.factorial(k)
                    * sm.factorial(l + m - k)
                    * sm.factorial(l - n - k)
                    * sm.factorial(n - m + k)
                )
            )
        return (
            sumterm
            * sm.exp(-sm.I * (alpha * n + gamma * m))
            * (-1) ** (n + m)
            * sm.sqrt(
                sm.factorial(l - m)
                * sm.factorial(l + m)
                * sm.factorial(l - n)
                * sm.factorial(l + n)
            )
        )

    def D(l, alpha, beta, gamma):
        res = sm.zeros(2 * l + 1, 2 * l + 1)
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                res[m + l, n + l] = Dmn(l, n, m, alpha, beta, gamma)
        return res

    def Umn(l, m, n):
        """Compute the (m, n) term of the transformation
        matrix from complex to real Ylms."""
        if n < 0:
            term1 = sm.I
        elif n == 0:
            term1 = sm.sqrt(2) / 2
        else:
            term1 = 1
        if (m > 0) and (n < 0) and (n % 2 == 0):
            term2 = -1
        elif (m > 0) and (n > 0) and (n % 2 != 0):
            term2 = -1
        else:
            term2 = 1
        return (
            term1
            * term2
            * 1
            / sm.sqrt(2)
            * (KroneckerDelta(m, n) + KroneckerDelta(m, -n))
        )

    def U(l):
        """Compute the U transformation matrix."""
        res = sm.zeros(2 * l + 1, 2 * l + 1)
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                res[m + l, n + l] = Umn(l, m, n)
        return res

    def REuler(l, alpha, beta, gamma):
        """Return the rotation matrix for a single degree `l`."""
        res = sm.zeros(2 * l + 1, 2 * l + 1)
        if l == 0:
            res[0, 0] = 1
            return res
        foo = sm.re(U(l).inv() * D(l, alpha, beta, gamma) * U(l))
        for m in range(2 * l + 1):
            for n in range(2 * l + 1):
                if abs(foo[m, n]) < 1e-15:
                    res[m, n] = 0
                else:
                    res[m, n] = foo[m, n]
        return res

    def RAxisAngle(l, u1, u2, u3, theta):
        """Axis-angle rotation matrix."""
        # Numerical tolerance
        tol = 1e-16
        if theta == 0:
            theta = tol
        if u1 == 0 and u2 == 0:
            u1 = tol
            u2 = tol

        # Elements of the transformation matrix
        costheta = sm.cos(theta)
        sintheta = sm.sin(theta)
        RA01 = u1 * u2 * (1 - costheta) - u3 * sintheta
        RA02 = u1 * u3 * (1 - costheta) + u2 * sintheta
        RA11 = costheta + u2 * u2 * (1 - costheta)
        RA12 = u2 * u3 * (1 - costheta) - u1 * sintheta
        RA20 = u3 * u1 * (1 - costheta) - u2 * sintheta
        RA21 = u3 * u2 * (1 - costheta) + u1 * sintheta
        RA22 = costheta + u3 * u3 * (1 - costheta)

        # Determine the Euler angles
        if (RA22 < -1 + tol) and (RA22 > -1 - tol):
            cosbeta = -1
            sinbeta = 0
            cosgamma = RA11
            singamma = RA01
            cosalpha = 1
            sinalpha = 0
        elif (RA22 < 1 + tol) and (RA22 > 1 - tol):
            cosbeta = 1
            sinbeta = 0
            cosgamma = RA11
            singamma = -RA01
            cosalpha = 1
            sinalpha = 0
        else:
            cosbeta = RA22
            sinbeta = sm.sqrt(1 - cosbeta**2)
            norm1 = sm.sqrt(RA20 * RA20 + RA21 * RA21)
            norm2 = sm.sqrt(RA02 * RA02 + RA12 * RA12)
            cosgamma = -RA20 / norm1
            singamma = RA21 / norm1
            cosalpha = RA02 / norm2
            sinalpha = RA12 / norm2
        alpha = sm.atan2(sinalpha, cosalpha)
        beta = sm.atan2(sinbeta, cosbeta)
        gamma = sm.atan2(singamma, cosgamma)

        return REuler(l, alpha, beta, gamma)

    def R(lmax, u, theta):
        """Return the full axis-angle rotation matrix up to degree `lmax`."""
        blocks = [RAxisAngle(l, *u, theta) for l in range(lmax + 1)]
        return sm.BlockDiagMatrix(*blocks)

    u = np.array(u, dtype=float)
    u /= np.sqrt(np.sum(u**2))
    return R(lmax, u, theta)
