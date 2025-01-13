from collections import defaultdict
from functools import reduce

import numpy as np

from jaxoplanet.starry.core import basis
from jaxoplanet.starry.multiprecision import mp, utils
from jaxoplanet.starry.multiprecision.utils import (
    fac as fac_function,
    kron_delta,
)

lmax = 20

FAC_CACHE = {}
CACHED_MATRICES = defaultdict(lambda: {})


def get_A(A, l_max, cache=None):
    if cache is None:
        cache = CACHED_MATRICES

    name = A.__name__.replace("_", "")
    if name not in cache[l_max]:
        print(f"pre-computing {name}...")
        cache[l_max][name] = A(l_max)
    return cache[l_max][name]


def fac(n):
    if n not in FAC_CACHE:
        FAC_CACHE[n] = fac_function(n)

    return FAC_CACHE[n]


def A(l, m):
    """A spherical harmonic normalization constant."""
    return mp.sqrt(
        (2 - kron_delta(m, 0)) * (2 * l + 1) * fac(l - m) / (4 * mp.pi * fac(l + m))
    )


def B(l, m, j, k):
    """Another spherical harmonic normalization constant."""

    ratio = fac((l + m + k - 1) / 2) / fac((-l + m + k - 1) / 2)

    return 2**l * fac(m) / (fac(j) * fac(k) * fac(m - j) * fac(l - m - k)) * ratio


def C(p, q, k):
    """Return the binomial theorem coefficient `C`."""
    return fac(k / 2) / (fac(q / 2) * fac((k - p) / 2) * fac((p - q) / 2))


def Y(l, m):
    """Return the spherical harmonic of degree `l` and order `m`."""
    coeffs = defaultdict(lambda: {})

    def get(function, *indices):
        fun_name = function.__name__
        if indices not in coeffs[fun_name]:
            coeffs[fun_name][indices] = function(*indices)

        return coeffs[fun_name][indices]

    res = defaultdict(lambda: 0)
    if m >= 0:
        for j in range(0, m + 1, 2):
            for k in range(0, l - m + 1, 2):
                for p in range(0, k + 1, 2):
                    for q in range(0, p + 1, 2):
                        coeff = (
                            (-1) ** ((j + p) // 2)
                            * get(A, l, m)
                            * get(B, l, m, j, k)
                            * get(C, p, q, k)
                        )
                        x_order = m - j + p - q
                        y_order = j + q
                        z_order = 0
                        res[(x_order, y_order, z_order)] += coeff
            for k in range(1, l - m + 1, 2):
                for p in range(0, k, 2):
                    for q in range(0, p + 1, 2):
                        coeff = (
                            (-1) ** ((j + p) // 2)
                            * get(A, l, m)
                            * get(B, l, m, j, k)
                            * get(C, p, q, k - 1)
                        )
                        x_order = m - j + p - q
                        y_order = j + q
                        z_order = 1
                        res[(x_order, y_order, z_order)] += coeff
    else:
        for j in range(1, abs(m) + 1, 2):
            for k in range(0, l - abs(m) + 1, 2):
                for p in range(0, k + 1, 2):
                    for q in range(0, p + 1, 2):
                        coeff = (
                            (-1) ** ((j + p - 1) // 2)
                            * get(A, l, abs(m))
                            * get(B, l, abs(m), j, k)
                            * get(C, p, q, k)
                        )
                        x_order = abs(m) - j + p - q
                        y_order = j + q
                        z_order = 0
                        res[(x_order, y_order, z_order)] += coeff
            for k in range(1, l - abs(m) + 1, 2):
                for p in range(0, k, 2):
                    for q in range(0, p + 1, 2):
                        coeff = (
                            (-1) ** ((j + p - 1) // 2)
                            * get(A, l, abs(m))
                            * get(B, l, abs(m), j, k)
                            * get(C, p, q, k - 1)
                        )
                        x_order = abs(m) - j + p - q
                        y_order = j + q
                        z_order = 1
                        res[(x_order, y_order, z_order)] += coeff

    return res


def p_coeffs(n):
    l = mp.floor(mp.sqrt(n))
    m = n - l * l - l
    mu = int(l - m)
    nu = int(l + m)
    if nu % 2 == 0:
        i = mu // 2
        j = nu // 2
        k = 0
    else:
        i = (mu - 1) // 2
        j = (nu - 1) // 2
        k = 1
    return (i, j, k)


def _A1(l_max):
    n = (l_max + 1) ** 2
    p = {m: p_coeffs(m) for m in range(n)}
    res = mp.zeros(n, n)

    k = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            y = Y(l, m)
            res[:, k] = mp.matrix([y[p[i]] for i in range(n)])
            k += 1

    res = res * 2 / mp.sqrt(mp.pi)
    return res


def A1(l_max, cache=None):
    if cache is None:
        cache = CACHED_MATRICES
    return get_A(_A1, l_max, cache=cache)


def gtilde(n):
    l = mp.floor(mp.sqrt(n))
    m = n - l * l - l
    mu = l - m
    nu = l + m
    if nu % 2 == 0:
        I = [mp.floor(mu / 2)]
        J = [mp.floor(nu / 2)]
        K = [0]
        C = [mp.floor((mu + 2) / 2)]
    elif (l == 1) and (m == 0):
        I = [0]
        J = [0]
        K = [1]
        C = [1]
    elif (mu == 1) and (l % 2 == 0):
        I = [l - 2]
        J = [1]
        K = [1]
        C = [3]
    elif mu == 1:
        I = [l - 3, l - 1, l - 3]
        J = [0, 0, 2]
        K = [1, 1, 1]
        C = [-1, 1, 4]
    else:
        I = [mp.floor((mu - 5) / 2), mp.floor((mu - 5) / 2), mp.floor((mu - 1) / 2)]
        J = [mp.floor((nu - 1) / 2), mp.floor((nu + 3) / 2), mp.floor((nu - 1) / 2)]
        K = [1, 1, 1]
        C = [mp.floor((mu - 3) / 2), -mp.floor((mu - 3) / 2), -mp.floor((mu + 3) / 2)]
    res = {}
    for i, j, k, c in zip(I, J, K, C, strict=False):
        res[(i, j, k)] = c
    return res


def A2_inv(l_max):
    n = (l_max + 1) ** 2
    p = {m: p_coeffs(m) for m in range(n)}
    res = mp.zeros(n, n)

    k = 0
    for l in range(l_max + 1):
        for _ in range(-l, l + 1):
            y = gtilde(k)
            res[:, k] = mp.matrix([y[p[i]] if p[i] in y else 0.0 for i in range(n)])
            k += 1

    return res


def _A2(lmax):
    """Compute the A2 matrix directly

    A2_inv is way faster to compute but then computing the inverse with mpmath is very
    slow (~4 min for lmax=20). This function makes the computation by solving the change
    basis matrix for each column of the A2_inv matrix.

    The way it works is to formalize the problem by building the A matrix which
    encapsulate the linear system. The tricky part is to find all possible powers to
    use, which is done recursively in the all_subsets_indices function.
    """
    n = (lmax + 1) ** 2
    A2 = mp.zeros(n, n)
    gs = [basis.gtilde(i) for i in range(n)]

    def all_subsets_indices(s, sets, indices=None):
        if indices is None:
            indices = []
        for si in s:
            for i, set_ in enumerate(sets):
                if si in set_ and i not in indices:
                    indices.append(i)
                    all_subsets_indices(set_, sets, indices)
        return indices

    for k in range(n):
        available_sets = gs[0 : int((np.floor(np.sqrt(k)) + 1) ** 2)]
        target_set = [basis.ptilde(k)]
        indices = np.array(all_subsets_indices(target_set, available_sets))

        powers = list(reduce(set.union, [set(gs[i].keys()) for i in indices]))

        A = mp.zeros(len(powers), len(indices))

        for i, ijk in enumerate(powers):
            for j, index in enumerate(indices):
                A[i, j] = gs[index].get(ijk, 0)

        b = np.array([target_set[0] == p for p in powers]).astype(int)

        # in multi-precision
        x = mp.lu_solve(A, utils.to_mp(b))

        for i, _x in zip(indices, x, strict=False):
            A2[k, i] += _x

    return A2.T


def A2(lmax, cache=None):
    if cache is None:
        cache = CACHED_MATRICES
    return get_A(_A2, lmax, cache=cache)
