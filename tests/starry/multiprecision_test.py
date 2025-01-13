import pytest


def test_l_precision():
    pytest.importorskip("mpmath")
    from jaxoplanet.starry.multiprecision import precision

    precision.starry_light_curve(2, 2, 0.1, 10)


def test_ld_precision():
    pytest.importorskip("mpmath")
    from jaxoplanet.starry.multiprecision import precision

    precision.limb_dark_light_curve(2, 0.1, 10)
