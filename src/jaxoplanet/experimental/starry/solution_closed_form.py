import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import binom, factorial, factorial2

from jaxoplanet.experimental.starry.cel import cel
from jaxoplanet.utils import get_dtype_eps

REFINE_J_AT = 25


def solution_vector(b, r):
    b = jnp.abs(b)
    r = jnp.abs(r)
    return _solution_vector(b, r)


@jax.custom_jvp
def _solution_vector(b, r):
    return _solution_vector_jvp((b, r), (jnp.zeros_like(b), jnp.zeros_like(r)))[0]


@_solution_vector.defjvp
def _solution_vector_jvp(primals, tangents):
    (b, r) = primals
    (b_dot, r_dot) = tangents

    tolerance = 10 * get_dtype_eps(b, r)
    (s0, s2), ((ds0db, ds0dr), (ds2db, ds2dr)) = _s0_s2_and_grad(b, r)
    s1, (ds1db, ds1dr) = _s1_and_grad(b, r)

    comp_occ = jnp.less_equal(b, r - 1)
    no_occ = jnp.logical_or(jnp.less(r, tolerance), jnp.greater_equal(b, r + 1))
    cond = jnp.logical_or(comp_occ, no_occ)

    s0 = jnp.where(no_occ, jnp.pi, jnp.where(comp_occ, 0, s0))
    s1 = jnp.where(no_occ, 2 * jnp.pi / 3, jnp.where(comp_occ, 0, s1))
    s2 = jnp.where(cond, 0, s2)

    ds0 = b_dot * jnp.where(cond, 0, ds0db) + r_dot * jnp.where(cond, 0, ds0dr)
    ds1 = b_dot * jnp.where(cond, 0, ds1db) + r_dot * jnp.where(cond, 0, ds1dr)
    ds2 = b_dot * jnp.where(cond, 0, ds2db) + r_dot * jnp.where(cond, 0, ds2dr)

    return (s0, s1, s2), (ds0, ds1, ds2)


@jax.custom_jvp
def _s0s2(b, r):
    return _s0_s2_and_grad(b, r)[0]


@_s0s2.defjvp
def _(primals, tangents):
    (b, r) = primals
    (b_dot, r_dot) = tangents
    (s0, s2), ((ds0db, ds0dr), (ds2db, ds2dr)) = _s0_s2_and_grad(b, r)
    return (s0, s2), ((b_dot * ds0db + r_dot * ds0dr), (b_dot * ds2db + r_dot * ds2dr))


def _s0_s2_and_grad(b, r):
    area, kappa0, kappa1 = kappas(b, r)

    b2 = jnp.square(b)
    r2 = jnp.square(r)
    r2pb2 = r2 + b2
    bpr = b + r
    onembpr2 = (1 + bpr) * (1 - bpr)
    eta2 = 0.5 * r2 * (r2 + 2 * b2)

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

    return (s0, s2), ((ds0db, ds0dr), (ds2db, ds2dr))


@jax.custom_jvp
def _s1(b, r):
    return _s1_and_grad(b, r)[0]


@_s1.defjvp
def _(primals, tangents):
    (b, r) = primals
    (b_dot, r_dot) = tangents
    s, (dsdb, dsdr) = _s1_and_grad(b, r)
    return s, b_dot * dsdb + r_dot * dsdr


def _s1_and_grad(b, r):

    # some pre-computations
    b2 = jnp.square(b)
    r2 = jnp.square(r)
    bpr = b + r
    bmr = b - r
    onembpr2 = (1 + bpr) * (1 - bpr)
    onembmr2 = (1 + bmr) * (1 - bmr)
    sqonembmr2 = jnp.sqrt(onembmr2)
    sqbr = jnp.sqrt(b * r)
    ratio = onembpr2 / onembmr2
    rootr1mr = jnp.sqrt(r * (1 - r))
    twodpi = 2 / jnp.pi

    # to deal with r = 0 or b = 0 none of the following variables are used
    fourbr = jnp.where(sqbr == 0.0, 1.0, 4 * b * r)
    invfourbr = 1 / fourbr
    bmrdbpr = bmr / bpr
    mu = 3 * bmrdbpr / onembmr2
    ksq = onembmr2 * invfourbr

    ksq_lt_1 = jnp.less(ksq, 1.0)
    kcsq = jnp.where(ksq_lt_1, -onembpr2 * invfourbr, ratio)

    k_arg = jnp.where(ksq_lt_1, ksq, 1 / ksq)
    p = jnp.where(ksq_lt_1, jnp.square(bmr), jnp.square(bmrdbpr)) * kcsq
    mk = 1 - kcsq

    a1 = jnp.where(ksq_lt_1, 0.0, 1 + mu)
    b1 = jnp.where(ksq_lt_1, 3 * kcsq * bmr * bpr, p + mu)
    Piofk = cel(k_arg, p, a1, b1)
    Eofk = cel(k_arg, 1.0, 1.0, kcsq)  # Agol 2019 (44)
    Em1mKdm = cel(k_arg, 1.0, 1.0, 0.0)  # Agol 2019 (45)

    # full condition list (see eq. 47 from Agol et al. 2019)
    cond_list = [
        r == 0.0,
        jnp.abs(r - b) >= 1.0,
        b == 0.0,
        (b == r) & (r == 0.5),
        (b == r) & (r < 0.5),
        (b == r) & (r > 0.5),
        b + r == 1.0,
        ksq < 1.0,
        ksq > 1.0,
    ]

    _L = [
        # r = 0
        0.0,
        # |r - b| >= 1
        0.0,
        # b = 0
        -2 * jnp.pi * (1 - r2) ** (3 / 2),
        # b = r = 1/2,
        jnp.pi - (4 / 3),
        # b = r < 1/2
        jnp.pi + (2 / 3) * cel(k_arg, 1.0, mk - 3.0, (1 - mk) * (2 * mk - 3)),
        # b = r > 1/2
        jnp.pi + (4 * r / 3) * cel(k_arg, 1.0, 1.0 - 3 * mk, mk - 1),
        # b + r = 1
        (
            2 * jnp.arccos(1.0 - 2 * r)
            - 4 * (3 + 2 * r - 8 * r2) * rootr1mr / 3
            - 2 * jnp.pi * (r > 0.5)
        ),
        # k^2 < 1
        onembmr2
        * (Piofk + (-3 + 6 * r2 + 2 * b * r) * Em1mKdm - fourbr * Eofk)
        / (3 * sqbr),
        # k^2 > 1
        2 * sqonembmr2 * (onembpr2 * Piofk - (4 - 7 * r2 - b2) * Eofk) / 3,
    ]

    L = jnp.select(cond_list, _L)

    # derivative w.r.t. r
    _dLdr = [
        # r = 0
        0.0,
        # |r - b| >= 1
        0.0,
        # b = 0
        2 * r * rootr1mr,
        # b = r = 1/2
        twodpi,
        # b = r < 1/2
        2 * twodpi * r * cel(k_arg, 1.0, 1.0, 4 * r2),
        # b = r > 1/2
        twodpi * Em1mKdm,
        # b + r = 1
        4 * twodpi * r / jnp.pi * rootr1mr,
        # k^2 < 1
        2 * r * onembmr2 * (-Em1mKdm + 2 * Eofk) / (3 * sqbr),
        # k^2 > 1
        -4 * r * sqonembmr2 * (Eofk - 2 * Em1mKdm) / 3,
    ]

    dLdr = jnp.select(cond_list, _dLdr)

    # derivative w.r.t. b
    _dLdb = [
        # r = 0
        0.0,
        # |r - b| >= 1
        0.0,
        # b = 0
        0.0,
        # b = r = 1/2
        -twodpi / 3,
        # b = r < 1/2
        2 * twodpi * r / 3 * cel(k_arg, 1.0, -1.0, -kcsq),
        # b = r > 1/2
        -2 * twodpi / 3 * cel(k_arg, 1.0, 1.0, 2 * kcsq),
        # b + r = 1
        -4 * twodpi / 3 * rootr1mr,
        # k^2 < 1
        -2 * r * onembmr2 * Em1mKdm / sqbr,
        # k^2 > 1
        -4 * r * sqonembmr2 * Eofk,
    ]

    dLdb = jnp.select(cond_list, _dLdb)

    return (2 * jnp.pi * (1 - (r > b)) - L) / 3, (dLdb, dLdr)


def kite_area(a, b, c):
    def sort2(a, b):
        return jnp.minimum(a, b), jnp.maximum(a, b)

    a, b = sort2(a, b)
    b, c = sort2(b, c)
    a, b = sort2(a, b)

    square_area = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return jnp.sqrt(jnp.maximum(0, square_area))  # We might not need this maximum


def kappas(b, r):
    # In this version, unlike the one in core/limb_dark.py, we don't need to
    # worry about zero/NaN-safety for the gradients since we also provide closed
    # form JVPs.
    b2 = jnp.square(b)
    factor = (r - 1) * (r + 1)
    area = kite_area(r, b, jnp.ones_like(r))
    return area, jnp.arctan2(area, b2 + factor), jnp.arctan2(area, b2 - factor)


# --------------------
# Closed form solution
# --------------------

from scipy.integrate import quad
from scipy.special import roots_legendre


def _J_numerical(v, k, order=100):
    """Return the integral J, evaluated numerically."""
    roots, weights = roots_legendre(order)
    kappa = jnp.where(k < 1, 2 * jnp.arcsin(k), jnp.pi)
    rng = 0.5 * kappa
    phi = rng * roots
    f = jnp.sin(phi) ** (2 * v) * (1 - k ** (-2) * jnp.sin(phi) ** 2) ** 1.5
    return rng * jnp.dot(f, weights)


def J_numerical(v, k):
    epsabs = 1e-15
    epsrel = 1e-13
    """Return the integral J, evaluated numerically."""
    kappa = 2 * np.arcsin(k) if k < 1 else np.pi
    func = lambda x: np.sin(x) ** (2 * v) * (1 - k ** (-2) * np.sin(x) ** 2) ** 1.5
    res, err = quad(func, -0.5 * kappa, 0.5 * kappa, epsabs=epsabs, epsrel=epsrel)
    return res


def A(i, u, v, delta):
    def _(i, u, v, j):
        return (
            binom(u, j)
            * binom(v, u + v - i - j)
            * (-1) ** (u + j)
            * delta ** (u + v - i - j)
        )

    bounds = np.array([np.max([[0, u - i]]), np.min([u + v - i, u]) + 1]).astype(int)
    return sum([_(i, u, v, j) for j in range(*bounds)])


def I_k2g1(v, I):
    if v not in I:
        I[v] = jnp.pi * factorial2(2 * v - 1) / (2**v * factorial(v))

    return I[v]


def J_down(v, k, k2, J):
    if v not in J:
        J[v] = (1 / ((2 * v + 1) * k2)) * (
            2 * (3 + v + (1 + v) * k2) * J_down(v + 1, k, k2, J)
            - (2 * v + 7) * J_down(v + 2, k, k2, J)
        )

    return J[v]


def J_up(v, k, k2, J):
    if v not in J:
        J[v] = (1 / (2 * v + 3)) * (
            2 * (v + (v - 1) * k2 + 1) * J_up(v - 1, k, k2, J)
            - k2 * (2 * v - 3) * J_up(v - 2, k, k2, J)
        )

    return J[v]


def I_k2l1_up(v, k, kc, I):
    if v not in I:
        I[v] = (1 / v) * (
            (2 * v - 1) / 2 * I_k2l1_up(v - 1, k, kc, I) - k ** (2 * v - 1) * kc
        )

    return I[v]


def IJ_k2l1_initial(k2):
    k = jnp.sqrt(k2)
    k3 = k2 * k
    mk = k2  # k^2 < 1
    Ek2 = cel(k2, 1.0, 1.0, 1 - mk)
    Kk2 = (Ek2 - mk * cel(k2, 1.0, 1.0, 0.0)) / (1 - mk)

    I = {0: 2 * jnp.arcsin(k)}
    J = {
        0: 2 / (3 * k3) * (2 * (2 * k2 - 1) * Ek2 + (1 - k2) * (2 - 3 * k2) * Kk2),
        1: 2
        / (15 * k3)
        * ((-3 * k2**2 + 13 * k2 - 8) * Ek2 + (1 - k2) * (8 - 9 * k2) * Kk2),
    }

    return I, J


def J_k2g1_Series(v, k, nterms=30):
    """Return the integral J, evaluated as a series."""
    res = 0
    for j in range(nterms):
        add = (
            (-1.0) ** j
            * binom(1.5, j)
            * factorial2(2 * j + 2 * v - 1)
            / (2 ** (j + v) * factorial(j + v))
            * k ** (-2 * j)
        )
        res += add
    return np.pi * float(res)


def IJ_k2g1_initial(k2, v_max=None):
    ik2 = 1 / k2
    mk = ik2  # k^2 >= 1
    Eik2 = cel(ik2, 1.0, 1.0, 1 - mk)
    Kik2 = (Eik2 - mk * cel(ik2, 1.0, 1.0, 0.0)) / (1 - mk)

    # Luger2019 - D51
    J = {
        0: (1 / 3) * ((8 - 4 * ik2) * Eik2 - 2 * (1 - ik2) * Kik2),
        1: (1 / 15)
        * ((-6 * k2 + 26 - 16 * ik2) * Eik2 + 2 * (1 - ik2) * (3 * k2 - 4) * Kik2),
    }

    if v_max is not None:
        n = (v_max // REFINE_J_AT) + 1
        for i in np.arange(REFINE_J_AT, REFINE_J_AT * n, REFINE_J_AT):
            J[i] = J_k2g1_Series(i, np.sqrt(k2))
            J[i - 1] = J_k2g1_Series(i - 1, np.sqrt(k2))

    return {}, J


def P_exact(l_max, b, r):
    b = jnp.abs(b)
    r = jnp.abs(r)
    k2 = jnp.maximum(0, (1 - r**2 - b**2 + 2 * b * r) / (4 * b * r))
    k = jnp.sqrt(k2)

    if k2 < 1.0:
        _I, _J = IJ_k2l1_initial(k2)
    else:
        _I = {}
        # The series expansion of J doesn't provide enough precision
        _J = {l_max: J_numerical(l_max, k), l_max - 1: J_numerical(l_max - 1, k)}

    def I(v):
        if v not in _I:
            if k2 < 1.0:
                _I[v] = I_k2l1_up(v, k, jnp.sqrt(1 - k2), _I)
            else:
                _I[v] = I_k2g1(v, _I)

        return _I[v]

    def J(v):
        if v not in _J:
            if k2 < 1.0:
                _J[v] = J_up(v, k, k2, _J)
            else:
                _J[v] = J_down(v, k, k2, _J)

        return _J[v]

    def K(u, v, delta):
        u = int(u)
        v = int(v)
        return np.sum([A(i, u, v, delta) * I(i + u) for i in range(u + v + 1)])

    def L(u, v, t, delta, k):
        u = int(u)
        v = int(v)
        return k**3 * np.sum(
            [A(i, u, v, delta) * J(i + u + t) for i in range(u + v + 1)]
        )

    # experimental, this is what we will ideally use if we figure
    # out the conditional logic
    def _P_green(r, b, l, m):
        r_cond = jnp.less(r, 10 * jnp.finfo(r.dtype).eps)
        delta = (b - r) / (2 * jnp.where(r_cond, 1, r))

        mu = l - m
        nu = l + m

        F = (2 * r) ** (l - 1) * (4 * b * r) ** 1.5

        conditions = [
            (mu / 2) % 2 == 0,
            (mu == 1) and (l % 2 == 0),
            (mu == 1) and (l != 1) and (l % 2 != 0),
            ((mu - 1) % 2) == 0 and ((mu - 1) // 2 % 2 == 0) and (l != 1),
            (mu == 1) and (l == 1),
        ]

        res = [
            2 * (2 * r) ** (l + 2) * K((mu + 4) / 4, nu / 2, delta),
            F * L((l - 2) / 2, 0, 0, delta, k) - 2 * L((l - 2) / 2, 0, 1, delta, k),
            F * L((l - 3) / 2, 1, 0, delta, k) - 2 * L((l - 3) / 2, 1, 1, delta, k),
            2 * F * L((mu - 1) / 4, (nu - 1) / 2, 0, delta, k),
            -1,
        ]

        return jnp.select(conditions, res, 0.0), 0

    def P_green(r, b, l, m):
        r_cond = jnp.less(r, 10 * jnp.finfo(r.dtype).eps)
        delta = (b - r) / (2 * jnp.where(r_cond, 1, r))

        mu = l - m
        nu = l + m

        F = (2 * r) ** (l - 1) * (4 * b * r) ** 1.5

        if mu == 1 and l == 1:
            case = 0
            P = -1

        elif mu % 2 == 0 and (mu // 2) % 2 == 0:
            case = 1
            P = 2 * (2 * r) ** (l + 2) * K((mu + 4) / 4, nu / 2, delta)

        elif mu == 1 and l % 2 == 0:
            case = 2
            P = F * (
                L((l - 2) / 2, 0, 0, delta, k) - 2 * L((l - 2) / 2, 0, 1, delta, k)
            )

        elif mu == 1:
            case = 3
            P = F * (
                L((l - 3) / 2, 1, 0, delta, k) - 2 * L((l - 3) / 2, 1, 1, delta, k)
            )

        elif (mu - 1) % 2 == 0 and ((mu - 1) // 2) % 2 == 0:
            case = 4
            P = 2 * F * L((mu - 1) / 4, (nu - 1) / 2, 0, delta, k)
        else:
            case = 5
            P = 0.0

        return P

    P = []

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            P.append(P_green(r, b, l, m))

    return jnp.array(P)
