# mypy: ignore-errors

import warnings

import jax
import numpy as np
import pytest

from jaxoplanet.core.limb_dark import solution_vector as solution_vector_limb
from jaxoplanet.experimental.starry.solution import _kappas, solution_vector
from jaxoplanet.experimental.starry.solution_closed_form import (
    solution_vector as solution_vector_closed,
)
from jaxoplanet.test_utils import assert_allclose


def test_kappas():
    def _phi(b, r):
        arg = 1 - r**2 - b**2
        m = (r > 0) & (b > 0)
        arg[m] /= 2 * r[m] * b[m]
        result = np.arcsin(np.clip(arg, -1, 1))
        result[~m] = 0.5 * np.pi * np.sign(arg[~m])
        return result

    def _lam(b, r):
        arg = 1 - r**2 + b**2
        m = (r > 0) & (b > 0)
        arg[m] /= 2 * b[m]
        result = np.arcsin(np.clip(arg, -1, 1))
        result[~m] = 0.5 * np.pi * np.sign(arg[~m])
        return result

    b, r = np.meshgrid(np.linspace(0, 2, 100), np.linspace(0.01, 1.5, 13))
    b = b.flatten()
    r = r.flatten()
    kappa0, kappa1 = jax.vmap(_kappas)(b, r)
    phi = _phi(b, r)
    lam = _lam(b, r)

    assert_allclose(kappa0, phi + 0.5 * np.pi)
    assert_allclose(kappa1, 0.5 * np.pi - lam)


@pytest.mark.parametrize("r", [0.1, 1.1])
def test_solution_compare_starry(r, l_max=10, order=20):
    starry = pytest.importorskip("starry")
    theano = pytest.importorskip("theano")
    theano.config.gcc__cxxflags += " -fexceptions"

    b = np.linspace(0, 1 + r, 501)[:-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = starry.Map(l_max)
        s_expect = m.ops.sT(b, r)
    s_calc = solution_vector(l_max, order=order)(b, r)

    for n in range(s_expect.shape[1]):
        # For logging/debugging purposes, work out the case id
        l = np.floor(np.sqrt(n)).astype(int)  # noqa
        m = n - l**2 - l
        mu = l - m
        nu = l + m
        if mu == 1 and l == 1:
            case = 1
        elif mu % 2 == 0 and (mu // 2) % 2 == 0:
            case = 2
        elif mu == 1 and l % 2 == 0:
            case = 3
        elif mu == 1:
            case = 4
        elif (mu - 1) % 2 == 0 and ((mu - 1) // 2) % 2 == 0:
            case = 5
        else:
            case = 0

        assert_allclose(
            s_calc[:, n],
            s_expect[:, n],
            err_msg=f"n={n}, l={l}, m={m}, mu={mu}, nu={nu}, case={case}",
            atol=1e-6,
        )


@pytest.mark.parametrize(
    "br",
    [
        (0.8, 0.3),  # "k2 < 1"
        (0.5, 0.3),  # "k2 > 1"
        (0.1, 0.1),  # "r = b < 1/2"
        (0.6, 0.6),  # "r = b > 1/2"
        (0.5, 0.5),  # "r = b = 1/2"
        (0.9, 0.1),  # "r + b = 1"
        (0.1, 2.0),  # "|r - b| >= 1"
        (0, 0.1),  # "b = 0"
        (0.1, 0.0),  # "r = 0"
    ],
)
def test_solution_closed_form(br):
    """Test the closed-form solution against the limb-darkened solution vector from
    Agol 2019."""
    b, r = br
    s_calc = solution_vector_limb(2, 500)(b, r)
    s_expect = solution_vector_closed(b, r)
    assert_allclose(s_calc, s_expect)
