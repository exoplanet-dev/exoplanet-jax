import numpy as np
import pytest

from jaxoplanet.starry.orbit import SurfaceBody
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.visualization import show_surface
from jaxoplanet.starry.ylm import Ylm


def test_show_map():
    _ = pytest.importorskip("matplotlib")
    y = Ylm.from_dense(np.hstack([1.0, np.random.rand(10) * 1e-1]))
    surface = Surface(y=y, inc=0.9, obl=-0.3, period=1.2, u=[0.5, 0.5])
    surface_body = SurfaceBody(period=1.0, surface=surface)

    show_surface(surface_body)
    show_surface(surface)
    show_surface(y)
