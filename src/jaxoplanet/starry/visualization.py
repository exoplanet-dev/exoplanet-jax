import numpy as np

from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.utils import graticule
from jaxoplanet.starry.ylm import Ylm


def show_surface(
    ylm_surface_body,
    theta: float = 0.0,
    res: int = 400,
    n: int = 6,
    ax=None,
    white_contour: bool = True,
    radius: float = None,
    include_phase: bool = True,
    **kwargs,
):
    """Show map of a

    Args:
        ylm_surface_body (Ylm, Surface or SurfaceBody): Ylm, Surface or Body with a
            surface
        theta (float, optional): Rotation angle of the map wrt its rotation axis.
            Defaults to 0.0.
        res (int, optional): Resolution of the map render. Defaults to 400.
        n (int, optional): number of latitude and longitude lines to show.
            Defaults to 6.
        ax (matplotlib.pyplot.Axes, optional): plot axes. Defaults to None.
        white_contour (bool, optional): Whether to surround the map by a white border
            (to hide border pixel aliasing). Defaults to True.
        radius (float, optional): Radius of the body. Defaults to None.
        include_phase (bool, optional): Whether to add the proper phase to the map to
            the rotation angle theta. Defaults to True.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
        if ax is None:
            ax = plt.subplot(111)

    if hasattr(ylm_surface_body, "surface"):
        surface = ylm_surface_body.surface
        if ylm_surface_body.radius is not None:
            radius = ylm_surface_body.radius.magnitude
        else:
            radius = 1.0 if radius is None else radius
        n = int(np.ceil(n * np.cbrt(radius)))
    # import Ylm leads to circular import
    elif isinstance(ylm_surface_body, Ylm):
        surface = Surface(y=ylm_surface_body)
        radius = 1.0 if radius is None else radius
    else:
        surface = ylm_surface_body
        radius = 1.0 if radius is None else radius

    phase = theta + (surface.phase if include_phase else 0.0)

    ax.imshow(
        surface.render(
            phase,
            res,
        ),
        origin="lower",
        **kwargs,
        extent=(-radius, radius, -radius, radius),
    )
    if n is not None:
        graticule(
            surface.inc,
            surface.obl,
            phase,
            radius=radius,
            n=n,
            white_contour=white_contour,
        )
    ax.axis(False)
