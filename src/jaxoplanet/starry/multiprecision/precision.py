import numpy as np

from jaxoplanet.core.limb_dark import light_curve as ld_light_curve
from jaxoplanet.starry import Surface, Ylm
from jaxoplanet.starry.light_curves import surface_light_curve
from jaxoplanet.starry.multiprecision import flux, mp


def starry_light_curve(
    ydeg: int, udeg: int, r: float, order: int, b: float | None = None
) -> float:
    """Estimate the absolute precision of a starry light curve.

    Args:
        ydeg (int): Degree of the spherical harmonics map.
        udeg (int): Order of the limb darkening polynomial.
        r (float): Radius ratio.
        order (int): Numerical integration order.
        b (float, optional): Impact parameter. Defaults to None.

    Returns:
        float: Precision of the jaxoplanet light curve model.
    """
    if b is None:
        b = 1 - r if r < 1.0 else r
    u = [1.0] * udeg
    y = np.ones((ydeg + 1) ** 2)
    surface = Surface(y=Ylm.from_dense(y), u=u, normalize=False)
    expected = flux.flux(ydeg=ydeg, udeg=udeg)(b, r, u=u, y=y)
    calc = surface_light_curve(surface, y=b, z=1.0, r=r, order=order)
    return float(np.abs(float(mp.mpf(float(calc)) - expected)))


def limb_dark_light_curve(
    udeg: int, r: float, order: int, b: float | None = None
) -> float:
    """Estimate the absolute precision of a limb darkened star light curve.

    Args:
        udeg (int): Order of the limb darkening polynomial.
        r (float): Radius ratio.
        order (int): Numerical integration order.
        b (float | None, optional): Impact parameter. Defaults to None.

    Returns:
        float: Precision of the jaxoplanet light curve model.
    """
    if b is None:
        b = 1 - r if r < 1.0 else r
    u = [1.0] * udeg
    expected = flux.flux(udeg=udeg)(b, r, u=u)
    calc = 1 + ld_light_curve(u, b, r, order=order)
    return np.abs(float(mp.mpf(float(calc)) - expected))
