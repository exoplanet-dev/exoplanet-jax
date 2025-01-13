from collections import defaultdict

from jaxoplanet.starry.multiprecision import mp

CACHED_MATRICES = defaultdict(
    lambda: {
        "sT": {},
    }
)


def get_sT(l_max, b, r, cache=None):
    if cache is None:
        cache = CACHED_MATRICES

    name = "sT"
    br = (b, r)
    if br not in cache[l_max][name]:
        cache[l_max][name][br] = sT(l_max, b, r)
    return cache[l_max][name][br]


def check_occultation(b, r):
    if not (abs(b) < r + 1.0):
        raise ValueError(f"No occultation with b = {b} and r = {r}.")
    if r >= 1 and b <= abs(1 - r):
        raise ValueError(f"occultation is full with b = {b} and r = {r}.")


def kappas(b, r):
    b2 = b**2
    factor = (r - 1) * (r + 1)
    area = kite_area(r, b, 1.0)
    return mp.atan2(area, b2 + factor), mp.atan2(area, b2 - factor)


def kite_area(a, b, c):
    def sort2(a, b):
        return min(a, b), max(a, b)

    a, b = sort2(a, b)
    b, c = sort2(b, c)
    a, b = sort2(a, b)

    square_area = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return mp.sqrt(max(0, square_area))


def P(l, m, b, r):
    """Compute the P integral numerically from its new parametrization."""
    mu = l - m
    nu = l + m

    if (abs(1 - r) < b) and (b < 1 + r):
        kappa = kappas(b, r)[0]
        phi = kappa - mp.pi / 2
    else:
        phi = mp.pi / 2
        kappa = mp.pi

    kappa = phi + mp.pi / 2
    delta = (b - r) / (2 * r)
    k2 = (1 - r**2 - b**2 + 2 * b * r) / (4 * b * r)

    if (mu / 2) % 2 == 0:

        def func(x):
            s = mp.sin(x)
            return (
                2
                * (2 * r) ** (l + 2)
                * (s**2 - s**4) ** (0.25 * (mu + 4))
                * (delta + s**2) ** (0.5 * nu)
            )

    elif (mu == 1) and (l % 2 == 0):

        def func(x):
            s = mp.sin(x)
            return (
                (2 * r) ** (l - 1)
                * (4 * b * r) ** (3.0 / 2.0)
                * (s**2 - s**4) ** (0.5 * (l - 2))
                * (k2 - s**2) ** (3.0 / 2.0)
                * (1 - 2 * s**2)
            )

    elif (mu == 1) and (l != 1) and (l % 2 != 0):

        def func(x):
            s = mp.sin(x)
            return (
                (2 * r) ** (l - 1)
                * (4 * b * r) ** (3.0 / 2.0)
                * (s**2 - s**4) ** (0.5 * (l - 3))
                * (delta + s**2)
                * (k2 - s**2) ** (3.0 / 2.0)
                * (1 - 2 * s**2)
            )

    elif ((mu - 1) % 2) == 0 and ((mu - 1) // 2 % 2 == 0) and (l != 1):

        def func(x):
            s = mp.sin(x)
            return (
                2
                * (2 * r) ** (l - 1)
                * (4 * b * r) ** (3.0 / 2.0)
                * (s**2 - s**4) ** (0.25 * (mu - 1))
                * (delta + s**2) ** (0.5 * (nu - 1))
                * (k2 - s**2) ** (3.0 / 2.0)
            )

    elif (mu == 1) and (l == 1):

        def func(x):
            c = mp.cos(x + 0.5 * kappa)
            omz2 = r**2 + b**2 - 2 * b * r * c
            if omz2 <= mp.mpf(0.0):
                return 0.0
            else:
                z2 = max(0, 1 - omz2)
                return 2 * r * (r - b * c) * (1 - z2 * mp.sqrt(z2)) / (3 * omz2)

    else:
        return 0
    res = mp.quad(func, [-kappa / 2, kappa / 2])
    return res


def p_numerical(l_max, b, r):
    """Compute the P integral numerically."""
    p = []

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            p.append(P(l, m, b, r))

    return mp.matrix(p).apply(mp.re)


def q_numerical(l_max, b, r):

    if (abs(1 - r) < b) and (b < 1 + r):
        lam = 0.5 * mp.pi - kappas(b, r)[1]
    else:
        lam = mp.pi / 2

    zero = 0.0
    c = mp.cos(lam)
    s = mp.sin(lam)
    h = {
        (0, 0): 2 * lam + mp.pi,
        (0, 1): -2 * c,
    }

    def get(u: int, v: int):
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
                U.append((mp.pi + 2 * lam) / 3)
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

    return mp.matrix(U)


def sT(l_max, b, r):
    r = abs(r)
    b = abs(b)
    return q_numerical(l_max, b, r) - p_numerical(l_max, b, r)


def rT(lmax: int):
    rt = [0.0 for _ in range((lmax + 1) * (lmax + 1))]
    amp0 = mp.pi
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

    amp0 = 0.5 * mp.pi
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
    return mp.matrix(rt)


def solution_limb_dark(degree, b, r):
    r2 = r**2
    b2 = b**2
    s = []

    k0, k1 = kappas(b, r)
    area = kite_area(r, b, 1.0)
    s0 = mp.pi - k1 - r2 * k0 + area * 0.5
    s.append(s0)

    if degree >= 1:

        def func(x):
            c = mp.cos(2 * x)
            omz2 = r2 + b2 - 2 * b * r * c
            if omz2 <= 0:
                return 0.0
            z2 = 1 - omz2
            z3 = mp.sqrt(z2) * z2
            return 2 * r * (r - b * c) * (1 - z3) / (3 * omz2)

        s1 = -mp.quad(func, [-k0 / 2, k0 / 2]) - 2 * (k1 - mp.pi) / 3

        s.append(s1.real)

    if degree >= 2:
        k2 = (1 - r2 - b2 + 2 * b * r) / (4 * b * r)

        if k2 <= 1:
            eta = (1 / (2 * mp.pi)) * (
                k1 + r2 * (r2 + 2 * b2) * k0 - 0.25 * (1 + 5 * r2 + b2) * area
            )
        elif k2 > 1:
            eta = 0.5 * r2 * (r2 + 2 * b2)

        s2 = 2 * s0 + 4 * mp.pi * eta - 2 * mp.pi
        s.append(s2)

    if degree >= 3:

        for n in range(3, degree + 1):

            def func(x):
                s2 = mp.sin(x) ** 2
                return (k2 - s2) ** (0.5 * n) * (r - b + 2 * b * s2)

            sn = 2 * r * (4 * b * r) ** (0.5 * n) * mp.quad(func, [-k0 / 2, k0 / 2])
            sn = sn.real

            s.append(-sn)

    return mp.matrix(s)
