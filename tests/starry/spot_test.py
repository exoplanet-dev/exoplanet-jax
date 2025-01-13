import jax
import numpy as np
import pytest

from jaxoplanet.starry import Surface
from jaxoplanet.starry.ylm import ylm_spot
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize("ydeg", [5, 10])
def test_spot_intensity(ydeg, radius=0.5, contrast=0.1):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False

    map = starry.Map(ydeg=ydeg)
    map.spot(contrast=contrast, radius=np.rad2deg(radius), lat=0.0, lon=0.0)

    lon = np.linspace(-np.pi, np.pi, 500)
    exp = map.intensity(0, np.rad2deg(lon))

    jax_map = Surface(y=ylm_spot(ydeg)(contrast, radius, 0.0, 0.0))
    calc = jax.vmap(jax_map.intensity, in_axes=(None, 0))(0.0, lon)

    assert_allclose(calc, exp)
