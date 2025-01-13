from collections.abc import Iterable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from jaxoplanet.starry.core.basis import A1, U, poly_basis
from jaxoplanet.starry.core.polynomials import Pijk
from jaxoplanet.starry.core.rotation import (
    full_rotation_axis_angle,
    left_project,
)
from jaxoplanet.starry.utils import ortho_grid
from jaxoplanet.starry.ylm import Ylm
from jaxoplanet.types import Array, Quantity


class Surface(eqx.Module):
    """Surface map object.

    Args:
        y (Optional(:py:class:`~jaxoplanet.starry.ylm.Ylm`)) Ylm object containing the
            spherical harmonic expansion of the map. Defaults to a uniform map with
            amplitude 1.0.
        inc (Optional[Quantity]): inclination of the map relative to line of sight.
            Defaults to pi/2 [angular unit].
        obl (Optional[Quantity]): obliquity of the map [angular unit]. Defaults to None.
        u (Optional[Array]): polynomial limb-darkening coefficients of the map.
        period (Optional[Quantity]): rotation period of the map [time unit]. Defaults to
            None.
        amplitude (Optional[float]): amplitude of the map; this quantity is proportional
            to the luminosity of the map and multiplies all flux-related observables.
            Defaults to 1.0.
        normalize (Optional(bool)): whether to normalize the coefficients of the
            spherical harmonics. If True, Ylm is normalized and the amplitude of the map
            is set to y[(0, 0)]. Defaults to True.
        phase (Optional[float]): initial phase of the map rotation around the polar
            axis. Defaults to 0.0.

    Example:

        .. code-block:: python

            import numpy as np
            import jax
            from jaxoplanet.starry.visualization import show_surface
            from jaxoplanet.starry.surface import Surface
            from jaxoplanet.starry.ylm import Ylm

            jax.config.update("jax_enable_x64", True)

            np.random.seed(30)
            y = Ylm.from_dense(np.random.rand(20))
            m = Surface(y=y, u=[0.5, 0.1], inc=0.9, obl=-0.3)
            show_surface(m)
    """

    y: Ylm
    """:py:class:`~starry.ylm.Ylm` object representing the spherical harmonic expansion
        of the map"""

    _inc: Array | None
    """Inclination of the map in radians. None if seen from the pole."""

    _obl: Array | None
    """Obliquity of the map in radians. None if no obliquity."""

    u: tuple[Array, ...]
    """Tuple of limb darkening coefficients."""

    period: Array | None
    """Rotation period of the map in days (attribute subject to change). None if not
    rotating."""

    amplitude: Array
    """Amplitude of the map, a quantity proportional to map luminosity."""

    normalize: bool
    """Boolean to specify whether the Ylm coefficients should be normalized"""

    phase: Array
    """Initial phase of the map rotation around polar axis"""

    def __init__(
        self,
        *,
        y: Ylm | None = None,
        inc: Quantity | None = 0.5 * jnp.pi,
        obl: Quantity | None = None,
        u: Iterable[Array] = (),
        period: Quantity | None = None,
        amplitude: Array = 1.0,
        normalize: bool = True,
        phase: Array = 0.0,
    ):

        if y is None:
            y = Ylm()

        if normalize:
            amplitude = jnp.array(y[(0, 0)], float)
            y = Ylm(data=y.data).normalize()

        self.y = y
        self._inc = inc
        self._obl = obl
        self.u = tuple(u)
        self.period = period
        self.amplitude = amplitude
        self.normalize = normalize
        self.phase = phase

    @property
    def inc(self):
        return self._inc if self._inc is not None else 0.0

    @inc.setter
    def inc(self, value):
        self._inc = value

    @property
    def obl(self):
        return self._obl if self._obl is not None else 0.0

    @obl.setter
    def obl(self, value):
        self._obl = value

    @property
    def _poly_basis(self):
        return jax.jit(poly_basis(self.deg))

    @property
    def udeg(self):
        """Order of the polynomial limb darkening."""
        return len(self.u)

    @property
    def ydeg(self):
        return self.y.deg

    @property
    def deg(self):
        """Total degree of the spherical harmonic expansion (``udeg + ydeg``)."""
        return self.ydeg + self.udeg

    def _intensity(self, x, y, z, theta=None):
        pT = self._poly_basis(x, y, z)
        Ry = left_project(self.ydeg, self.inc, self.obl, theta, 0.0, self.y.todense())
        A1Ry = A1(self.ydeg).todense() @ Ry
        p_y = Pijk.from_dense(A1Ry, degree=self.ydeg)
        u = jnp.array([1, *self.u])
        p_u = Pijk.from_dense(u @ U(self.udeg), degree=self.udeg)
        p = (p_y * p_u).todense()
        return pT @ p * self.amplitude

    @partial(jax.jit, static_argnames=("res",))
    def render(self, theta: float | None = None, res: int = 400):
        """Returns the intensity map projected onto the x-y plane (sky).

        Args:
            theta (float, optional): rotation angle of the map. Defaults to 0.0.
            res (int, optional): resolution of the render. Defaults to 400.

        Returns:
            ArrayLike: square 2D array representing the intensity map
            (with nans outside the map disk).
        """
        _, xyz = ortho_grid(res)
        intensity = self._intensity(*xyz, theta=theta)
        return jnp.reshape(intensity, (res, res))

    def intensity(self, lat: float, lon: float):
        """Returns the intensity of the map at a given latitude and longitude.

        Args:
            lat (float): latitude in the rest frame of the map
            lon (float): longitude in the rest frame of the map

        Returns:
            float: intensity of the map at the given latitude and longitude
        """
        lon = lon + jnp.pi / 2  # convention, 0 lon faces the observer
        lat = jnp.pi / 2 - lat  # convention, latitude 0 is equator
        x = jnp.sin(lat) * jnp.cos(lon)
        y = jnp.sin(lat) * jnp.sin(lon)
        z = jnp.cos(lat) * jnp.ones_like(x)

        axis = full_rotation_axis_angle(self.inc - jnp.pi / 2, self.obl, 0.0, 0.0)
        axis = jnp.array(axis[0:3]) * axis[-1]
        rotation = Rotation.from_rotvec(axis)
        x, y, z = rotation.apply(jnp.array([x, y, z]).T).T

        return self._intensity(x, y, z)

    def rotational_phase(self, time: Array) -> Array | None:
        """Returns the rotational phase of the map at a given time.

        Args:
            time (ArrayLike): time in same units as the period

        Returns:
            ArrayLike: rotational phase of the map at the given time
        """
        if self.period is None:
            return None
        else:
            return 2 * jnp.pi * time / self.period + self.phase
