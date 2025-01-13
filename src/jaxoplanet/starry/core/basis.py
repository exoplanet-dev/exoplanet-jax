import math
from collections import defaultdict
from functools import partial

import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg
from jax.experimental.sparse import BCOO
from scipy.special import comb, gamma

try:
    from scipy.sparse import csc_array
except ImportError:
    # With older versions of scipy, the data structures were called "matrices"
    # not "arrays"; this allows us to support either.
    from scipy.sparse import csc_matrix as csc_array


def basis(lmax):
    """Full change of basis matrix from spherical harmonics to Green's basis

    Args:
        lmax (int): maximum degree of the spherical harmonic basis
    """
    matrix = scipy.sparse.linalg.spsolve(A2_inv(lmax), A1(lmax))
    if lmax > 0:
        return BCOO.from_scipy_sparse(matrix)
    else:
        return BCOO.fromdense(np.squeeze(matrix)[None, None])


def A1(lmax):
    """Change of basis matrix from spherical harmonics to polynomial basis.

    Args:
        lmax (int): Maximum degree of the spherical harmonic basis.

    Returns:
        TODO: Description of the return value.
    """
    return _A_impl(lmax, p_Y) * 2 / np.sqrt(np.pi)


def A2_inv(lmax):
    """Change of basis matrix from polynomial basis to Green's basis.

    Args:
        lmax (int): Maximum degree of the spherical harmonic basis.

    Returns:
        TODO: Description of the return value.
    """
    return _A_impl(lmax, p_G)


def _A_impl(lmax, func):
    """Return a sparse change of basis matrix given a function that maps
    to the polynomial basis.

    Args:
        lmax (int): Maximum degree of the spherical harmonic basis.
        func (callable): Function that maps to the polynomial basis (signature
        irrelevant here and used for convenience). The output must be a tuple
        of (indices, data) where indices is a list of indices of the polynomial basis
        terms and data is a list of the coefficients of the polynomial basis terms
        (see `p_Y` and `p_G`).

    Returns:
        _type_: _description_
    """
    n = (lmax + 1) ** 2
    data = []
    row_ind = []
    col_ind = []
    p = {ptilde(m): m for m in range(n)}
    n = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            idx, val = func(p, l, m, n)
            data.extend(val)
            row_ind.extend(idx)
            col_ind.extend([n] * len(idx))
            n += 1
    return csc_array((np.array(data), (row_ind, col_ind)), shape=(n, n))


def ptilde(n):
    """Compute the x, y, and z powers of the n-th polynomial basis term.

    If the n-th term is x^i y^j z^k, return (i, j, k).

    Args:
        n (int): Index of the polynomial basis term.

    Returns:
        tuple: (i, j, k)

    Example:
        >>> ptilde(2) # z
        (0, 0, 1)

        >>> ptilde(3) # x + y
        (1, 1, 0)
    """
    l = math.floor(math.sqrt(n))
    m = n - l * l - l
    mu = l - m
    nu = l + m
    if nu % 2 == 0:
        i = mu // 2
        j = nu // 2
        k = 0
    else:
        i = (mu - 1) // 2
        j = (nu - 1) // 2
        k = 1
    return (i, j, k)


def Alm(l, m):
    return math.sqrt(
        (2 - int(m == 0))
        * (2 * l + 1)
        * math.factorial(l - m)
        / (4 * math.pi * math.factorial(l + m))
    )


def Blmjk(l, m, j, k):
    a = l + m + k - 1
    b = -l + m + k - 1
    if (b < 0) and (b % 2 == 0):
        return 0
    else:
        ratio = gamma(0.5 * a + 1) / gamma(0.5 * b + 1)
    return (
        2**l
        * math.factorial(m)
        / (
            math.factorial(j)
            * math.factorial(k)
            * math.factorial(m - j)
            * math.factorial(l - m - k)
        )
        * ratio
    )


def Cpqk(p, q, k):
    return math.factorial(k // 2) / (
        math.factorial(q // 2)
        * math.factorial((k - p) // 2)
        * math.factorial((p - q) // 2)
    )


def Ylm(l, m):
    """Compute the coefficients of the spherical harmonic Y_{l,m}.

    Args:
        l (int): Degree of the spherical harmonic.
        m (int): Order of the spherical harmonic in the range [-l, l].

    Returns:
        dict: {(i, j, k): coeff} where i, j, k are the powers of x, y, z (see `ptilde`).

    Example:
        >>> Ylm(2, 0)
        {(0, 0, 0): 0.6307831305050402,
            (2, 0, 0): -0.9461746957575603,
            (0, 2, 0): -0.9461746957575603}
    """
    res = defaultdict(lambda: 0)
    A = Alm(l, abs(m))
    for j in range(int(m < 0), abs(m) + 1, 2):
        for k in range(0, l - abs(m) + 1, 2):
            B = Blmjk(l, abs(m), j, k)
            if not B:
                continue
            factor = A * B
            for p in range(0, k + 1, 2):
                for q in range(0, p + 1, 2):
                    ind = (abs(m) - j + p - q, j + q, 0)
                    res[ind] += (
                        (-1) ** ((j + p - (m < 0)) // 2) * factor * Cpqk(p, q, k)
                    )
        for k in range(1, l - abs(m) + 1, 2):
            B = Blmjk(l, abs(m), j, k)
            if not B:
                continue
            factor = A * B
            for p in range(0, k, 2):
                for q in range(0, p + 1, 2):
                    ind = (abs(m) - j + p - q, j + q, 1)
                    res[ind] += (
                        (-1) ** ((j + p - (m < 0)) // 2) * factor * Cpqk(p, q, k - 1)
                    )

    return dict(res)


def p_Y(p, l, m, n):
    """Return a representation of Y_{l, m} in the polynomial basis.

    Args:
        p (dict): Powers of xyz as returned by `ptilde`.
        l (int): Degree of the spherical harmonic.
        m (int): Order of the spherical harmonic in the range [-l, l].
        n (None): Dummy variable.

    Returns:
        tuple: (indices, data) where indices is an np.array of indices of the polynomial
            basis terms and data is an np.array of the coefficients of the polynomial
            basis terms.

    Example:
        >>> p = {ptilde(m): m for m in range(9)}
        >>> p_Y(p, 2, 0, 0)
        (array([0, 4, 8]), array([ 0.63078313, -0.9461747 , -0.9461747 ]))
        # see correspondence with `Ylm` example
    """
    del n
    indices = []
    data = []
    for k, v in Ylm(l, m).items():
        if k not in p:
            continue
        indices.append(p[k])
        data.append(v)
    indices = np.array(indices, dtype=int)
    data = np.array(data, dtype=float)
    idx = np.argsort(indices)
    return indices[idx], data[idx]


def gtilde(n):
    """Compute the n-th term of the Green basis in the polynomial basis.

    Args:
        n (int): Index of the Green basis term.

    Returns:
        dict: {(i, j, k): coeff} where i, j, k are the powers of x, y, z (see `ptilde`).

    Example:
        >>> gtilde(50)
        {(4, 0, 1): 5, (4, 2, 1): -5, (6, 0, 1): -8}
    """
    l = math.floor(math.sqrt(n))
    m = n - l * l - l
    mu = l - m
    nu = l + m
    if nu % 2 == 0:
        I = [mu // 2]
        J = [nu // 2]
        K = [0]
        C = [(mu + 2) // 2]
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
        I = [(mu - 5) // 2, (mu - 5) // 2, (mu - 1) // 2]
        J = [(nu - 1) // 2, (nu + 3) // 2, (nu - 1) // 2]
        K = [1, 1, 1]
        C = [(mu - 3) // 2, -(mu - 3) // 2, -(mu + 3) // 2]
    res = {}
    for i, j, k, c in zip(I, J, K, C, strict=False):
        if c != 0.0:
            res[(i, j, k)] = c
    return res


def p_G(p, l, m, n):
    """Return a representation of Green's basis n-th term in the polynomial basis.

    Args:
        p (dict): Powers of xyz as returned by `ptilde`.
        l (None): Dummy variable.
        m (None): Dummy variable.
        n (int): Index of the Green basis term.

    Returns:
        tuple: (indices, data) where indices is an np.array of indices of the polynomial
            basis terms and data is an np.array of the coefficients of the polynomial
            basis terms.

    Example:
        >>> p = {ptilde(n): n for n in range(100)}
        >>> p_G(p, None, None, 50)
        (array([26, 50, 54]), array([ 5., -8., -5.]))
    """
    del l, m
    indices = []
    data = []
    for k, v in gtilde(n).items():
        if k not in p:
            continue
        indices.append(p[k])
        data.append(v)
    indices = np.array(indices, dtype=int)
    data = np.array(data, dtype=float)
    idx = np.argsort(indices)
    return indices[idx], data[idx]


def poly_basis(deg):
    N = (deg + 1) * (deg + 1)

    @partial(jnp.vectorize, signature=f"(),(),()->({N})")
    def impl(x, y, z):
        xarr = [None for _ in range(N)]
        yarr = [None for _ in range(N)]

        # Ensures we get `nan`s off the disk
        xterm = 1.0 + 0.0 * z
        yterm = 1.0 + 0.0 * z

        i0 = 0
        di0 = 3
        j0 = 0
        dj0 = 2
        for n in range(deg + 1):
            i = i0
            di = di0
            xarr[i] = xterm
            j = j0
            dj = dj0
            yarr[j] = yterm
            i = i0 + di - 1
            j = j0 + dj - 1
            while i + 1 < N:
                xarr[i] = xterm
                xarr[i + 1] = xterm
                di += 2
                i += di
                yarr[j] = yterm
                yarr[j + 1] = yterm
                dj += 2
                j += dj - 1
            xterm *= x
            i0 += 2 * n + 1
            di0 += 2
            yterm *= y
            j0 += 2 * (n + 1) + 1
            dj0 += 2

        assert all(v is not None for v in xarr)
        assert all(v is not None for v in yarr)

        inds = []
        n = 0
        for ell in range(deg + 1):
            for m in range(-ell, ell + 1):
                if (ell + m) % 2 != 0:
                    inds.append(n)
                n += 1

        p = jnp.array(xarr) * jnp.array(yarr)
        if len(inds):
            return p.at[np.array(inds)].multiply(z)
        else:
            return p

    return impl


def utilde(n):
    res = defaultdict(float)

    if n == 0:
        return {(0, 0, 0): 1.0}

    for k in range(n + 1):
        c1 = comb(n, k) * (-1) ** k
        k2 = k // 2
        for j in range(k2 + 1):
            c2 = comb(k2, j) * (-1) ** j
            for l in range(j + 1):
                c3 = comb(j, l)
                res[(2 * (j - l), 2 * l, k % 2)] += -c1 * c2 * c3

    return res


def u_p(p, l, m, n):
    # this is very similar to gtilde, might be a more general
    # way of doing the _A_impl function without dummy variables
    del l, m
    indices = []
    data = []
    for k, v in utilde(n).items():
        if k not in p:
            continue
        indices.append(p[k])
        data.append(v)
    indices = np.array(indices, dtype=int)
    data = np.array(data, dtype=float)
    idx = np.argsort(indices)
    return indices[idx], data[idx]


def U(udeg: int):
    """Change of basis matrix from limb darkening basis to polynomial basis.

    Args:
        udeg (int): Degree of the limb darkening basis.

    Returns:
        TODO
    """
    n = (udeg + 1) ** 2
    p = {ptilde(m): m for m in range(n)}
    P = np.zeros((udeg + 1, n))
    for i in range(udeg + 1):
        idxs, values = u_p(p, None, None, i)
        for j, v in zip(idxs, values, strict=False):
            P[i, j] += v

    return P
