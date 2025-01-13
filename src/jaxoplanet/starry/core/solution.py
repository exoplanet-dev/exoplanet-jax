from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

from jaxoplanet.core.limb_dark import kite_area
from jaxoplanet.types import Array
from jaxoplanet.utils import zero_safe_sqrt


def solution_vector(l_max: int, order: int = 20) -> Callable[[Array, Array], Array]:
    n_max = l_max**2 + 2 * l_max + 1

    @jax.jit
    @partial(jnp.vectorize, signature=f"(),()->({n_max})")
    def impl(b: Array, r: Array) -> Array:
        b = jnp.abs(b)
        r = jnp.abs(r)
        kappa0, kappa1 = kappas(b, r)
        P = p_integral(order, l_max, b, r, kappa0)
        Q = q_integral(l_max, 0.5 * jnp.pi - kappa1)
        return Q - P

    return impl


def kappas(b: Array, r: Array) -> tuple[Array, Array]:
    b2 = jnp.square(b)
    factor = (r - 1) * (r + 1)
    b_cond = jnp.logical_and(jnp.greater(b, jnp.abs(1 - r)), jnp.less(b, 1 + r))
    b_ = jnp.where(b_cond, b, 1)
    area = jnp.where(b_cond, kite_area(r, b_, 1), 0)
    return jnp.arctan2(area, b2 + factor), jnp.arctan2(area, b2 - factor)


def q_integral(l_max: int, lam: Array) -> Array:
    zero = jnp.zeros_like(lam)
    c = jnp.cos(lam)
    s = jnp.sin(lam)
    h = {
        (0, 0): 2 * lam + jnp.pi,
        (0, 1): -2 * c,
    }

    def get(u: int, v: int) -> Array:
        if (u, v) in h:
            return h[(u, v)]
        if u >= 2:
            comp = 2 * c ** (u - 1) * s ** (v + 1) + (u - 1) * get(u - 2, v)
        else:
            assert v >= 2
            comp = -2 * c ** (u + 1) * s ** (v - 1) + (v - 1) * get(u, v - 2)
        comp /= u + v
        h[(u, v)] = comp
        return comp

    U = []
    for l in range(l_max + 1):  # noqa
        for m in range(-l, l + 1):
            if l == 1 and m == 0:
                U.append((np.pi + 2 * lam) / 3)
                continue
            mu = l - m
            nu = l + m
            if (mu % 2) == 0 and (mu // 2) % 2 == 0:
                u = mu // 2 + 2
                v = nu // 2
                assert u % 2 == 0
                U.append(get(u, v))
            else:
                U.append(zero)

    return jnp.stack(U)


def p_integral(order: int, l_max: int, b: Array, r: Array, kappa0: Array) -> Array:
    """Numerical integration of the P integral using the Gauss-Legendre quadrature.

    As described in Equation D32 of Luger et al. (2019), there are 6 cases to consider.
    Empirically, we notice that the numerical integration of the first case (mu/2 even)
    is precise at very low order. Hence ``low_order=30``is used for the first case. For
    the other cases, we use the order specified by the user, renamed in the function
    ``high_order``. We also note that outside the linear limb-darkening case (i.e.
    (l,m)=(1, 0), or n=2) the integrand is symmetrical in phi, so we can evaluate the
    integral over half the range and multiply by 2.


    Parameters
    ----------
    order : int
        The order of the Gauss-Legendre quadrature.
    l_max : int
        The maximum degree of the spherical harmonic expansion.
    b : Array
        Impact parameter.
    r : Array
        Occultor radius.
    kappa0 : Array
        k0 angle.

    Returns
    -------
    Array
        The integral of the P function over the occultor surface.
    """
    b2 = jnp.square(b)
    r2 = jnp.square(r)

    # This is a hack for when r -> 0 or b -> 0, so k2 -> inf
    factor = 4 * b * r
    k2_cond = jnp.less(factor, 10 * jnp.finfo(factor.dtype).eps)
    factor = jnp.where(k2_cond, 1, factor)
    k2 = jnp.maximum(0, (1 - r2 - b2 + 2 * b * r) / factor)
    # And for when r -> 0
    r_cond = jnp.less(r, 10 * jnp.finfo(r.dtype).eps)
    delta = (b - r) / (2 * jnp.where(r_cond, 1, r))
    rng = 0.25 * kappa0

    # low order variables
    low_order = np.min([order, 20])
    zeros = jnp.zeros(order - low_order)
    roots, low_weights = roots_legendre(low_order)
    low_weights = jnp.hstack((low_weights, zeros))
    phi = rng * (roots + 1)
    low_s2 = jnp.square(jnp.sin(phi))
    low_a1 = low_s2 - jnp.square(low_s2)
    low_a2 = jnp.where(r_cond, 0, delta + low_s2)

    # high order variables
    high_order = order
    high_roots, high_weights = roots_legendre(high_order)
    phi = rng * (high_roots + 1)
    high_s2 = jnp.square(jnp.sin(phi))
    high_f0 = jnp.maximum(0, jnp.where(k2_cond, 1 - r2, factor * (k2 - high_s2))) ** 1.5
    high_a1 = high_s2 - jnp.square(high_s2)
    high_a2 = jnp.where(r_cond, 0, delta + high_s2)
    high_a4 = 1 - 2 * high_s2

    indices = []
    weights = []
    integrand = []
    n = 0

    for l in range(l_max + 1):  # noqa
        high_fa3 = (2 * r) ** (l - 1) * high_f0
        for m in range(-l, l + 1):
            mu = l - m
            nu = l + m

            if mu == 1 and l == 1:
                phi = 2 * rng * high_roots
                c = jnp.cos(phi + 0.5 * kappa0)
                omz2 = r2 + b2 - 2 * b * r * c
                cond = jnp.less(omz2, 10 * jnp.finfo(omz2.dtype).eps)
                omz2 = jnp.where(cond, 1, omz2)
                z2 = jnp.maximum(0, 1 - omz2)
                result = (
                    2 * r * (r - b * c) * (1 - z2 * zero_safe_sqrt(z2)) / (3 * omz2)
                )
                integrand.append(jnp.where(cond, 0, 2 * result))
                weights.append(high_weights)

            elif mu % 2 == 0 and (mu // 2) % 2 == 0:
                f = (
                    2
                    * (2 * r) ** (l + 2)
                    * low_a1 ** (0.25 * (mu + 4))
                    * low_a2 ** (0.5 * nu)
                )
                integrand.append(2 * jnp.hstack((f, zeros)))
                weights.append(low_weights)

            elif mu == 1 and l % 2 == 0:
                f = high_fa3 * high_a1 ** (l // 2 - 1) * high_a4
                integrand.append(2 * f)
                weights.append(high_weights)

            elif mu == 1:
                f = high_fa3 * high_a1 ** ((l - 3) // 2) * high_a2 * high_a4
                integrand.append(2 * f)
                weights.append(high_weights)

            elif (mu - 1) % 2 == 0 and ((mu - 1) // 2) % 2 == 0:
                f = (
                    2
                    * high_fa3
                    * high_a1 ** ((mu - 1) // 4)
                    * high_a2 ** (0.5 * (nu - 1))
                )
                integrand.append(2 * f)
                weights.append(high_weights)

            else:
                n += 1
                continue

            indices.append(n)
            n += 1

    indices = np.stack(indices)
    weights = jnp.stack(weights)

    P0 = rng * jnp.sum(jnp.stack(integrand) * weights, axis=1)
    P = jnp.zeros(l_max**2 + 2 * l_max + 1)

    return P.at[indices].set(P0)


def rT(lmax: int) -> Array:
    rt = [0.0 for _ in range((lmax + 1) * (lmax + 1))]
    amp0 = jnp.pi
    lfac1 = 1.0
    lfac2 = 2.0 / 3.0
    for ell in range(0, lmax + 1, 4):
        amp = amp0
        for m in range(0, ell + 1, 4):
            mu = ell - m
            nu = ell + m
            rt[ell * ell + ell + m] = amp * lfac1
            rt[ell * ell + ell - m] = amp * lfac1
            if ell < lmax:
                rt[(ell + 1) * (ell + 1) + ell + m + 1] = amp * lfac2
                rt[(ell + 1) * (ell + 1) + ell - m + 1] = amp * lfac2
            amp *= (nu + 2.0) / (mu - 2.0)
        lfac1 /= (ell / 2 + 2) * (ell / 2 + 3)
        lfac2 /= (ell / 2 + 2.5) * (ell / 2 + 3.5)
        amp0 *= 0.0625 * (ell + 2) * (ell + 2)

    amp0 = 0.5 * jnp.pi
    lfac1 = 0.5
    lfac2 = 4.0 / 15.0
    for ell in range(2, lmax + 1, 4):
        amp = amp0
        for m in range(2, ell + 1, 4):
            mu = ell - m
            nu = ell + m
            rt[ell * ell + ell + m] = amp * lfac1
            rt[ell * ell + ell - m] = amp * lfac1
            if ell < lmax:
                rt[(ell + 1) * (ell + 1) + ell + m + 1] = amp * lfac2
                rt[(ell + 1) * (ell + 1) + ell - m + 1] = amp * lfac2
            amp *= (nu + 2.0) / (mu - 2.0)
        lfac1 /= (ell / 2 + 2) * (ell / 2 + 3)
        lfac2 /= (ell / 2 + 2.5) * (ell / 2 + 3.5)
        amp0 *= 0.0625 * ell * (ell + 4)
    return np.array(rt)
