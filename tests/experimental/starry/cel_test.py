import pytest
import scipy.special as sps
from jax.test_util import check_grads

from jaxoplanet.experimental.starry.cel import cel
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize("k", [-0.1, 0.0, 0.1, 0.5, 0.9, 0.99])
def test_cel_ellipk(k):
    m = k**2
    computed = cel(m, 1.0, 1.0, 1.0)
    expected = sps.ellipk(m)
    assert_allclose(computed, expected)


@pytest.mark.parametrize("k", [-0.1, 0.0, 0.1, 0.5, 0.9, 0.99])
def test_cel_ellipe(k):
    m = k**2
    computed = cel(m, 1.0, 1.0, 1.0 - m)
    expected = sps.ellipe(m)
    assert_allclose(computed, expected)


@pytest.mark.parametrize("k", [-0.1, 0.0, 0.1, 0.5, 0.9])
@pytest.mark.parametrize("p", [-5.0, -1.0, -0.5, 0.5, 0.8])
def test_cel_grads(k, p):
    m = k**2
    check_grads(cel, (m, p, 1.0, 1.0), order=1)
