from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

from jaxoplanet.types import Array


class Pijk(eqx.Module):
    r"""A class to represent and manipulate spherical harmonics in the
    polynomial basis. Several indices are used throughout the class:

    * Indices :math:`(i, j, k)` represent the order of the polynomials of
      :math:`(x, y, z)`, for example :math:`(1, 0, 2)` represents :math:`x\,z^2`.

    * Indices :math:`(l, m)` represent the orders of the spherical harmonics.

    * Index n represent the index of the polynomial in the flattened array.

    Flattened array ``todense`` and ``from_dense`` follow the convention from
    Luger et al. (2019). More specifically:

    .. math::

        \tilde{p} =
        \begin{pmatrix}
            1 & x & y & z & x^2 & xz & xy & yz & y^2 &
            \cdot\cdot\cdot
        \end{pmatrix}^\mathsf{T}

    """

    data: dict[tuple[int, int, int], Array]
    degree: int = eqx.field(static=True)
    diagonal: bool = eqx.field(static=True)

    def __init__(self, data: Mapping[tuple[int, int, int], Array], degree: int = None):
        self.data = dict(data)
        self.degree = (
            max(self.ijk2lm(*ijk)[0] for ijk in data.keys())
            if degree is None
            else degree
        )
        self.diagonal = all(self.ijk2lm(*ijk)[1] == 0.0 for ijk in data.keys())

    @property
    def shape(self):
        return (self.degree + 1) ** 2

    @property
    def indices(self):
        return self.data.keys()

    @staticmethod
    def ijk2lm(i, j, k):
        assert k in (0, 1)

        if k == 0:
            nu = 2 * j
            mu = 2 * i
        else:
            nu = 2 * j + 1
            mu = 2 * i + 1

        m = int((nu - mu) / 2)
        l = int((nu + mu) / 2)

        return l, m

    @staticmethod
    def n2lm(n):
        l = int(np.floor(np.sqrt(n)))
        m = n - l**2 - l
        return l, m

    @staticmethod
    def lm2n(l, m):
        return l**2 + l + m

    @staticmethod
    def lm2ijk(l, m):
        mu = l - m
        nu = l + m

        if nu % 2 == 0:
            return mu // 2, nu // 2, 0
        else:
            return (mu - 1) // 2, (nu - 1) // 2, 1

    @staticmethod
    def n2ijk(n):
        return Pijk.lm2ijk(*Pijk.n2lm(n))

    def index(self, i, j, k):
        l, m = self.ijk2lm(i, j, k)
        return int((l + 1) * l + m)

    def tosparse(self) -> BCOO:
        indices, values = zip(*self.data.items(), strict=False)
        idx = np.array([self.index(i, j, k) for i, j, k in indices])[:, None]
        return BCOO((jnp.asarray(values), idx), shape=(self.shape,))

    def todense(self) -> Array:
        return self.tosparse().todense()

    @classmethod
    def from_dense(cls, x: Array, degree: int = None) -> "Pijk":
        data = defaultdict(float)
        for i in range(len(x)):
            data[Pijk.n2ijk(i)] = x[i]

        return cls(data, degree=degree)

    def __add__(self, other: "Pijk") -> "Pijk":
        if isinstance(other, Pijk):
            return _add(self, other)
        else:
            return jax.tree_util.tree_map(lambda x: x + other, self)

    def __mul__(self, other: Any) -> "Pijk":
        if isinstance(other, Pijk):
            return _mul(self, other)
        else:
            return jax.tree_util.tree_map(lambda x: x * other, self)

    def __rmul__(self, other: Any) -> "Pijk":
        assert not isinstance(other, Pijk)
        return jax.tree_util.tree_map(lambda x: other * x, self)

    def __getitem__(self, key) -> Array:
        assert isinstance(key, tuple)
        return self.todense()[self.index(*key)]


def _add(u: Pijk, v: Pijk) -> Pijk:
    upv = defaultdict(float)
    for (i1, j1, k1), v1 in u.data.items():
        upv[(i1, j1, k1)] += v1 + v.data.get((i1, j1, k1), 0.0)

    for (i2, j2, k2), v2 in v.data.items():
        if (i2, j2, k2) not in upv:
            upv[(i2, j2, k2)] += v2 + u.data.get((i2, j2, k2), 0.0)

    return Pijk(upv)


def _mul(u: Pijk, v: Pijk) -> Pijk:
    uv = defaultdict(float)
    for (i1, j1, k1), v1 in u.data.items():
        for (i2, j2, k2), v2 in v.data.items():
            v1v2 = v1 * v2
            if (k1 + k2) == 2:
                uv[(i1 + i2, j1 + j2, 0)] += v1v2
                uv[(i1 + i2 + 2, j1 + j2, 0)] -= v1v2
                uv[(i1 + i2, j1 + j2 + 2, 0)] -= v1v2
            else:
                uv[(i1 + i2, j1 + j2, k1 + k2)] += v1v2

    return Pijk(uv)


def polynomial_product_matrix(p, degree):
    """Given a polynomial vector p, return a matrix M such that M @ p2 is the polynomial
    product of p with p2.

    Note: This function was implemented to reproduce the filter matrix from starry.
    However, tests show that using the polynomial multiplication operator on the class
    is faster than introducing a matrix multiplication (see surface_light_curve.py).

    Args:
        p (Array): A vector in the polynomial basis (in its dense form).
        degree (int): Degree of the polynomial to be multiplied with.

    Returns:
        Array: The polynomial product matrix.
    """
    p = jnp.array(p)
    deg_p = int(np.floor(np.sqrt(len(p))))
    M = jnp.zeros(
        ((deg_p + degree + 1) * (deg_p + degree + 1), (degree + 1) * (degree + 1))
    )
    n1 = 0

    for l1 in range(degree + 1):
        for m1 in range(-l1, l1 + 1):
            odd1 = (l1 + m1) % 2 != 0
            n2 = 0
            for l2 in range(deg_p + 1):
                for m2 in range(-l2, l2 + 1):
                    l = l1 + l2
                    n = l * l + l + m1 + m2
                    if odd1 and (l2 + m2) % 2 != 0:
                        M = M.at[n - 4 * l + 2, n1].set(M[n - 4 * l + 2, n1] + p[n2])
                        M = M.at[n - 2, n1].set(M[n - 2, n1] - p[n2])
                        M = M.at[n + 2, n1].set(M[n + 2, n1] - p[n2])
                    else:
                        M = M.at[n, n1].set(M[n, n1] + p[n2])
                    n2 += 1
            n1 += 1

    return M
