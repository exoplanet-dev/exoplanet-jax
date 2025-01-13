import pytest

from jaxoplanet.starry.ylm import Ylm
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize("data", [{(5, 0): 5.0, (3, -1): 1.5}])
def test_starry_ylm_compare(data):
    starry = pytest.importorskip("starry")
    theano = pytest.importorskip("theano")
    theano.config.gcc__cxxflags += " -fexceptions"
    starry.config.lazy = False

    data = dict(data)
    data.update({(0, 0): 1.0})
    ylm = Ylm(data)
    starry_map = starry.Map(ylm.deg)
    for (ell, m), c in data.items():
        if (ell, m) == (0, 0):
            continue
        starry_map[ell, m] = c
    expected = starry_map._y
    calc = ylm.todense()
    assert_allclose(expected, calc)
