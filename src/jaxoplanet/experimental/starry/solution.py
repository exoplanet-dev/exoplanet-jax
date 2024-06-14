from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

from jaxoplanet.experimental.starry.cel import cel
from jaxoplanet.types import Array
from jaxoplanet.utils import get_dtype_eps, zero_safe_sqrt


def solution_vector(l_max: int, order: int = 20) -> Callable[[Array, Array], Array]:
    n_max = l_max**2 + 2 * l_max + 1

    @jax.jit
    @partial(jnp.vectorize, signature=f"(),()->({n_max})")
    def impl_numerical(b: Array, r: Array) -> Array:
        b = jnp.abs(b)
        r = jnp.abs(r)
        kappa0, kappa1 = _kappas(b, r)
        P = _p_integral_numerical(order, l_max, b, r, kappa0)
        Q = _q_integral_numerical(l_max, 0.5 * jnp.pi - kappa1)
        return Q - P

    return impl_numerical


def _kite_area(a: Array, b: Array, c: Array) -> Array:
    def sort2(a: Array, b: Array) -> tuple[Array, Array]:
        return jnp.minimum(a, b), jnp.maximum(a, b)

    a, b = sort2(a, b)
    b, c = sort2(b, c)
    a, b = sort2(a, b)

    square_area = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return zero_safe_sqrt(jnp.maximum(0, square_area))


# ------------------
# Numerical solution
# ------------------


def _kappas(b: Array, r: Array) -> tuple[Array, Array]:
    b2 = jnp.square(b)
    factor = (r - 1) * (r + 1)
    b_cond = jnp.logical_and(jnp.greater(b, jnp.abs(1 - r)), jnp.less(b, 1 + r))
    b_ = jnp.where(b_cond, b, 1)
    area = jnp.where(b_cond, _kite_area(r, b_, 1), 0)
    return jnp.arctan2(area, b2 + factor), jnp.arctan2(area, b2 - factor)


def _q_integral_numerical(l_max: int, lam: Array) -> Array:
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


def _p_integral_numerical(
    order: int, l_max: int, b: Array, r: Array, kappa0: Array
) -> Array:
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

    roots, weights = roots_legendre(order)
    rng = 0.5 * kappa0
    phi = rng * roots
    c = jnp.cos(phi + 0.5 * kappa0)
    s = jnp.sin(phi)
    s2 = jnp.square(s)

    f0 = jnp.maximum(0, jnp.where(k2_cond, 1 - r2, factor * (k2 - s2))) ** 1.5
    a1 = s2 - jnp.square(s2)
    a2 = jnp.where(r_cond, 0, delta + s2)
    a4 = 1 - 2 * s2

    ind = []
    arg = []
    n = 0
    for l in range(l_max + 1):  # noqa
        fa3 = (2 * r) ** (l - 1) * f0
        for m in range(-l, l + 1):
            mu = l - m
            nu = l + m

            if mu == 1 and l == 1:
                omz2 = r2 + b2 - 2 * b * r * c
                cond = jnp.less(omz2, 10 * jnp.finfo(omz2.dtype).eps)
                omz2 = jnp.where(cond, 1, omz2)
                z2 = jnp.maximum(0, 1 - omz2)
                result = 2 * r * (r - b * c) * (1 - z2 * jnp.sqrt(z2)) / (3 * omz2)
                arg.append(jnp.where(cond, 0, result))

            elif mu % 2 == 0 and (mu // 2) % 2 == 0:
                arg.append(
                    2 * (2 * r) ** (l + 2) * a1 ** (0.25 * (mu + 4)) * a2 ** (0.5 * nu)
                )

            elif mu == 1 and l % 2 == 0:
                arg.append(fa3 * a1 ** (l // 2 - 1) * a4)

            elif mu == 1:
                arg.append(fa3 * a1 ** ((l - 3) // 2) * a2 * a4)

            elif (mu - 1) % 2 == 0 and ((mu - 1) // 2) % 2 == 0:
                arg.append(2 * fa3 * a1 ** ((mu - 1) // 4) * a2 ** (0.5 * (nu - 1)))

            else:
                n += 1
                continue

            ind.append(n)
            n += 1

    P0 = rng * jnp.sum(jnp.stack(arg) * weights[None, :], axis=1)
    P = jnp.zeros(l_max**2 + 2 * l_max + 1)

    # Yes, using np not jnp here: 'ind' is always static.
    inds = np.stack(ind)

    return P.at[inds].set(P0)


# --------------------
# Closed form solution
# --------------------


def _closed_form_and_grad(b, r):
    area, kappa0, kappa1 = _kappas(b, r)

    b2 = jnp.square(b)
    r2 = jnp.square(r)
    r2pb2 = r2 + b2
    bpr = b + r
    onembpr2 = (1 + bpr) * (1 - bpr)
    eta2 = 0.5 * r2 * (r2 + 2 * b2)

    # ********* #
    # s0 and s2 #
    # ********* #

    # ksq > 1
    s0_lrg = jnp.pi * (1 - r2)
    offset_lrg = 4 * jnp.pi * (eta2 - 0.5)

    # ksq <= 1
    Alens = kappa1 + r2 * kappa0 - area * 0.5
    s0_sml = jnp.pi - Alens
    offset_sml = 2 * (
        -(jnp.pi - kappa1) + 2 * eta2 * kappa0 - 0.25 * area * (1 + 5 * r2 + b2)
    )

    delta = 4 * b * r
    cond = jnp.greater(onembpr2 + delta, delta)
    s0 = jnp.where(cond, s0_lrg, s0_sml)
    s2 = 2 * s0 + jnp.where(cond, offset_lrg, offset_sml)

    ds0db = jnp.where(cond, 0, area / b)
    ds0dr = -2 * r * jnp.where(cond, jnp.pi, kappa0)

    ds2db = 2 * ds0db + jnp.where(
        cond,
        8 * jnp.pi * b * r2,
        2 * (4 * b2 * r2 * kappa0 - (1 + r2pb2) * area) / b,
    )
    ds2dr = 2 * ds0dr + jnp.where(
        cond, 8 * jnp.pi * r * r2pb2, 8 * r * (r2pb2 * kappa0 - area)
    )

    # ** #
    # s1 #
    # ** #

    # TODO(dfm): Handle the following edge cases:
    #            - (b >= 1 + r) || (r == 0): no occultation
    #            - b <= r - 1: full occultation
    #            - b == 0
    #            - b == r
    #
    # Perhaps these should be handled at the top level calling function?

    # TODO(dfm): Check this tolerance and add test
    tolerance = 10 * get_dtype_eps(b, r)

    b2 = jnp.square(b)
    r2 = jnp.square(r)
    bpr = b + r
    bmr = b - r
    onembpr2 = (1 + bpr) * (1 - bpr)
    onembmr2 = (1 + bmr) * (1 - bmr)
    sqonembmr2 = jnp.sqrt(onembmr2)
    sqbr = jnp.sqrt(b * r)
    ratio = onembpr2 / onembmr2
    fourbr = 4 * b * r
    invfourbr = 1 / fourbr
    bmrdbpr = bmr / bpr
    mu = 3 * bmrdbpr / onembmr2

    ksq = onembpr2 * invfourbr + 1
    ksq_cond = jnp.less(ksq, 1.0)
    kcsq = jnp.where(ksq_cond, -onembpr2 * invfourbr, ratio)

    k_arg = jnp.where(ksq_cond, ksq, 1 / ksq)
    p = jnp.where(ksq_cond, jnp.square(bmr), jnp.square(bmrdbpr)) * kcsq

    a1 = jnp.where(ksq_cond, 0.0, 1 + mu)
    b1 = jnp.where(ksq_cond, 3 * kcsq * bmr * bpr, p + mu)
    Piofk = cel(k_arg, p, a1, b1)
    Eofk = cel(k_arg, 1.0, 1.0, kcsq)
    Em1mKdm = cel(k_arg, 1.0, 1.0, 0.0)

    lambda1 = jnp.where(
        ksq_cond,
        onembmr2
        * (Piofk + (-3 + 6 * r2 + 2 * b * r) * Em1mKdm - fourbr * Eofk)
        / (3 * sqbr),
        2 * sqonembmr2 * (onembpr2 * Piofk - (4 - 7 * r2 - b2) * Eofk) / 3,
    )

    dsdb = jnp.where(
        ksq_cond,
        2 * r * onembmr2 * (-Em1mKdm + 2 * Eofk) / (3 * sqbr),
        -4 * r * sqonembmr2 * (Eofk - 2 * Em1mKdm) / 3,
    )
    dsdr = jnp.where(
        ksq_cond, -2 * r * onembmr2 * Em1mKdm / sqbr, -4 * r * sqonembmr2 * Eofk
    )

    # Below we handle the edge cases:

    # 1. ksq ~ 1
    ksq_one = jnp.less(jnp.abs(kcsq), tolerance)
    rootr1mr = jnp.sqrt(r * (1 - r))
    lambda1 = jnp.where(
        ksq_one,
        2 * jnp.arccos(1.0 - 2.0 * r)
        - 4 * (3 + 2 * r - 8 * r2) * rootr1mr / 3
        - 2 * jnp.pi * (r > 0.5),
        lambda1,
    )
    tmp = 8 * r * rootr1mr
    ds1db = jnp.where(ksq_one, -tmp, dsdb)
    ds1dr = jnp.where(ksq_one, tmp / 3, dsdr)
    s1 = (2 * jnp.pi * (1 - (r > b)) - lambda1) / 3

    return (s0, s1, s2), ((ds0db, ds0dr), (ds1db, ds1dr), (ds2db, ds2dr))


def _wallis(n: int):
    """Return (something like the) Wallis ratio,

        Gamma(1 + n / 2) / Gamma(3 / 2 + n / 2)

    Computes it recursively. Using double precision, the error is below 1e-14
    well past n = 100.
    """
    if n % 2 == 0:
        z = 1 + n // 2
        dz = -1
    else:
        z = 1 + (n - 1) // 2
        dz = 0

    A = 1.0
    B = jnp.sqrt(jnp.pi)
    for i in range(z + dz):
        A *= i + 1
        B *= i - 0.5
    for i in range(max(1, z + dz), z + 1):
        B *= i - 0.5

    if n % 2 == 0:
        return A / B
    else:
        return B / A


def _compute_m_coeff(lmax, max_iter):
    M_coeff = np.empty((4, max_iter))
    for j in range(4):
        n = lmax - 3 + j
        coeff = np.sqrt(np.pi) * _wallis(n)
        M_coeff[j, 0] = coeff
        for i in range(1, max_iter):
            coeff *= (2 * i - 1) ** 2 / (2 * i * (1 + n + 2 * i))
            M_coeff[j, i] = coeff
    return M_coeff


def _compute_n_coeff(lmax, max_iter):
    N_coeff = np.empty((2, max_iter))
    for j in range(2):
        n = lmax - 1 + j
        coeff = np.sqrt(np.pi) * _wallis(n) / (n + 3)
        N_coeff[j, 0] = coeff
        for i in range(1, max_iter):
            coeff *= (4 * i**2 - 1) / (2 * i * (3 + n + 2 * i))
            N_coeff[j, i] = coeff
    return N_coeff


def _downward_m(lmax, max_iter, k, ksq, sqonembmr2):
    tol = get_dtype_eps(ksq)
    M_coeff = _compute_m_coeff(lmax, max_iter)
    M = [0 for _ in range(lmax)]
    fac = k * sqonembmr2 ** (lmax - 3)

    for j in range(4):

        def cond(args):
            (n, val) = args
            return abs(val) > tol

        def body(args):
            (
                n,
                val,
                k2n,
            ) = args

        val = M_coeff[j, 0]
        k2n = 1.0
        for n in range(1, max_iter):
            k2n *= ksq
            term = k2n * M_coeff(j, n)
            val += term
            if abs(term) < tol:
                break

        M[lmax - 3 + j] = val * fac
        fac *= sqonembmr2
    # for (int j = 0; j < 4; ++j) {
    #   // Add leading term to M
    #   val = M_coeff(j, 0);
    #   k2n = 1.0;

    #   // Compute higher order terms
    #   for (int n = 1; n < STARRY_MN_MAX_ITER; ++n) {
    #     k2n *= ksq;
    #     term = k2n * M_coeff(j, n);
    #     val += term;
    #     if (abs(term) < tol)
    #       break;
    #   }
    #   M(lmax - 3 + j) = val * fac;
    #   fac *= sqonembmr2;
    # }


#   T val, k2n, tol, fac, term;
#   T invsqarea = T(1.0) / sqarea;

#   // Compute highest four using a series solution
#   if (ksq < 1) {
#     // Compute leading coefficient (n=0)
#     tol = mach_eps<T>() * ksq;
#     term = 0.0;
#     fac = 1.0;
#     for (int n = 0; n < lmax - 3; ++n)
#       fac *= sqonembmr2;
#     fac *= k;

#     // Now, compute higher order terms until
#     // desired precision is reached
#     for (int j = 0; j < 4; ++j) {
#       // Add leading term to M
#       val = M_coeff(j, 0);
#       k2n = 1.0;

#       // Compute higher order terms
#       for (int n = 1; n < STARRY_MN_MAX_ITER; ++n) {
#         k2n *= ksq;
#         term = k2n * M_coeff(j, n);
#         val += term;
#         if (abs(term) < tol)
#           break;
#       }
#       M(lmax - 3 + j) = val * fac;
#       fac *= sqonembmr2;
#     }
