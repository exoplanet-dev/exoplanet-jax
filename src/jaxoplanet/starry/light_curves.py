from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet.starry.core.basis import A1, A2_inv, U
from jaxoplanet.starry.core.polynomials import Pijk
from jaxoplanet.starry.core.rotation import left_project
from jaxoplanet.starry.core.solution import rT, solution_vector
from jaxoplanet.starry.orbit import SurfaceSystem
from jaxoplanet.starry.surface import Surface
from jaxoplanet.types import Array, Quantity
from jaxoplanet.units import quantity_input, unit_registry as ureg


def light_curve(
    system: SurfaceSystem, order: int = 20
) -> Callable[[Quantity], tuple[Array | None, Array | None]]:
    """Return a function that computes the light curve of a system made of non-uniform
       surfaces.

    Args:
        system (SurfaceSystem): system object
        order (int, optional): order of the P integral numerical approximation.
            Defaults to 20.

    Returns:
        Callable[[Quantity], tuple[Array | None, Array | None]]: a function that computes
        the separate light curves for each body, starting with the central body and
        ordered by the bodies list (from first to last inserted).
    """
    central_bodies_lc = jax.vmap(
        surface_light_curve, in_axes=(None, 0, 0, 0, 0, None, None)
    )

    @partial(system.surface_vmap, in_axes=(0, 0, 0, 0, None))
    def compute_body_light_curve(surface, radius, x, y, z, time):
        if surface is None:
            return 0.0
        else:
            theta = surface.rotational_phase(time.magnitude)
            return surface_light_curve(
                surface,
                (system.central.radius / radius).magnitude,
                (x / radius).magnitude,
                (y / radius).magnitude,
                (z / radius).magnitude,
                theta,
                order,
            )

    @quantity_input(time=ureg.day)
    @vectorize
    def light_curve_impl(time: Quantity) -> Array:
        if system.central_surface is None:
            central_light_curves = jnp.array([0.0])
        else:
            theta = system.central_surface.rotational_phase(time.magnitude)
            central_radius = system.central.radius
            central_phase_curve = surface_light_curve(
                system.central_surface, theta=theta, order=order
            )
            if len(system.bodies) > 0:
                xos, yos, zos = system.relative_position(time)
                n = len(xos.magnitude)
                central_light_curves = central_bodies_lc(
                    system.central_surface,
                    (system.radius / central_radius).magnitude,
                    (xos / central_radius).magnitude,
                    (yos / central_radius).magnitude,
                    (zos / central_radius).magnitude,
                    theta,
                    order,
                )

                if n > 1 and central_light_curves is not None:
                    central_light_curves = central_light_curves.sum(
                        0
                    ) - central_phase_curve * (n - 1)
                    central_light_curves = jnp.expand_dims(central_light_curves, 0)

                body_light_curves = compute_body_light_curve(
                    system.radius, -xos, -yos, -zos, time
                )

                return jnp.hstack([central_light_curves, body_light_curves])
            else:
                return jnp.array([central_phase_curve])

    return light_curve_impl


# TODO: figure out the sparse matrices (and Pijk) to avoid todense()
def surface_light_curve(
    surface: Surface,
    r: float | None = None,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    theta: float | None = None,
    order: int = 20,
    higher_precision: bool = False,
):
    """Light curve of an occulted surface.

    Args:
        surface (Surface): Surface object
        r (float or None): radius of the occulting body, relative to the current map
           body
        x (float or None): x coordinate of the occulting body relative to the surface
           center. By default (None) 0.0
        y (float or None): y coordinate of the occulting body relative to the surface
           center. By default (None) 0.0
        z (float or None): z coordinate of the occulting body relative to the surface
           center. By default (None) 0.0
        theta (float):
            rotation angle of the map, in radians. By default 0.0
        order (int):
            order of the P integral numerical approximation. By default 20
        higher_precision (bool): whether to compute change of basis matrix as hight
            precision. By default False (only used to testing).

    Returns:
        ArrayLike: flux
    """
    if higher_precision:
        try:
            from jaxoplanet.starry.multiprecision import (
                basis as basis_mp,
                utils as utils_mp,
            )
        except ImportError as e:
            raise ImportError(
                "The `mpmath` Python package is required for higher_precision=True."
            ) from e

    rT_deg = rT(surface.deg)

    x = 0.0 if x is None else x
    y = 0.0 if y is None else y
    z = 0.0 if z is None else z

    # no occulting body
    if r is None:
        b_rot = True
        theta_z = 0.0
        design_matrix_p = rT_deg

    # occulting body
    else:
        b = jnp.sqrt(jnp.square(x) + jnp.square(y))
        b_rot = jnp.logical_or(jnp.greater_equal(b, 1.0 + r), jnp.less_equal(z, 0.0))
        b_occ = jnp.logical_not(b_rot)
        theta_z = jnp.arctan2(x, y)

        # trick to avoid nan `x=jnp.where...` grad caused by nan sT
        r = jnp.where(b_rot, 1.0, r)
        b = jnp.where(b_rot, 1.0, b)

        sT = solution_vector(surface.deg, order=order)(b, r)

        if surface.deg > 0:
            if higher_precision:
                A2 = np.atleast_2d(utils_mp.to_numpy(basis_mp.A2(surface.deg)))
            else:
                A2 = scipy.sparse.linalg.inv(A2_inv(surface.deg))
                A2 = jax.experimental.sparse.BCOO.from_scipy_sparse(A2)
        else:
            A2 = jnp.array([[1]])

        design_matrix_p = jnp.where(b_occ, sT @ A2, rT_deg)

    if surface.ydeg == 0:
        rotated_y = surface.y.todense()
    else:
        rotated_y = left_project(
            surface.ydeg,
            surface._inc,
            surface._obl,
            theta,
            theta_z,
            surface.y.todense(),
        )

    # limb darkening
    if surface.udeg == 0:
        p_u = Pijk.from_dense(jnp.array([1]))
    else:
        u = jnp.array([1, *surface.u])
        p_u = Pijk.from_dense(u @ U(surface.udeg), degree=surface.udeg)

    # surface map * limb darkening map
    if higher_precision:
        A1_val = np.atleast_2d(utils_mp.to_numpy(basis_mp.A1(surface.ydeg)))
    else:
        A1_val = jax.experimental.sparse.BCOO.from_scipy_sparse(A1(surface.ydeg))

    p_y = Pijk.from_dense(A1_val @ rotated_y, degree=surface.ydeg)
    p_y = p_y * p_u

    norm = np.pi / (p_u.tosparse() @ rT(surface.udeg))

    return surface.amplitude * (p_y.tosparse() @ design_matrix_p) * norm
