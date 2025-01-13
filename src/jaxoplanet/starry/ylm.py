r"""A module to manipulate vectors in the spherical harmonic basis.

The spherical harmonics basis is a set of orthogonal functions defined on the
unit sphere. In jaxoplanet, this basis is used to represent the intensity at the surface
of a spherical body, such as a star or a planet. We say that :math:`y` represents the
intensity of a surface in the spherical harmonics basis if the specific intensity at the
:math:`(x,y)` on the surface can be written as:

.. math::

    I(x, y) = \mathbf{\tilde{y}_n^\mathsf{T}} (x, y) \, \mathbf{y}
    \quad,

where :math:`\tilde{y}_n` is the **spherical harmonic basis**,
arranged in increasing degree and order:

.. math::

    \mathbf{\tilde{y}_n} =
    \begin{pmatrix}
        Y_{0, 0} &
        Y_{1, -1} & Y_{1, 0} & Y_{1, 1} &
        Y_{2, -2} & Y_{2, -1} & Y_{2, 0} & Y_{2, 1} & Y_{2, 2} &
        \cdot\cdot\cdot
    \end{pmatrix}^\mathsf{T}
    \quad,

where :math:`Y_{l, m} = Y_{l, m}(x, y)` is the spherical harmonic of degree :math:`l`
and order :math:`m`. For reference, in this basis the coefficient of the spherical
harmonic :math:`Y_{l, m}` is located at the index

.. math::

    n = l^2 + l + m

"""

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from scipy.special import legendre as LegendreP

from jaxoplanet.starry.core import basis, solution
from jaxoplanet.starry.core.polynomials import Pijk
from jaxoplanet.starry.core.rotation import dot_rotation_matrix
from jaxoplanet.starry.core.wigner3j import Wigner3jCalculator
from jaxoplanet.types import Array


class Ylm(eqx.Module):
    """Ylm object containing the spherical harmonic coefficients.

    Args:
        data (Mapping[tuple[int, int], Array], optional): dictionary of
            spherical harmonic coefficients. Defaults to {(0, 0): 1.0}.
    """

    data: dict[tuple[int, int], Array]
    """coefficients of the spherical harmonic expansion of the map in the form
    `{(l, m): coefficient}`"""

    deg: int = eqx.field(static=True)
    """The maximum degree of the spherical harmonic coefficients."""

    diagonal: bool = eqx.field(static=True)
    """Whether are orders m of the spherical harmonic coefficients are zero.
    Diagonal if only the degrees "l" are non-zero."""

    def __init__(
        self,
        data: Mapping[tuple[int, int], Array] | None = None,
    ):
        if data is None:
            data = {(0, 0): 1.0}

        self.data = dict(data)
        self.deg = max(ell for ell, _ in data.keys())
        self.diagonal = all(m == 0 for _, m in data.keys())

    @property
    def shape(self) -> tuple[int, ...]:
        """The number of coefficients in the basis. This sets the shape of
        the output of `todense`."""
        return (self.deg**2 + 2 * self.deg + 1,)

    @property
    def indices(self) -> list[tuple[int, int]]:
        """List of (l,m) indices of the spherical harmonic coefficients."""
        return list(self.data.keys())

    @staticmethod
    def index(l: Array, m: Array) -> Array:
        """Convert the degree and order of the spherical harmonic to the
        corresponding index in the coefficient array."""
        return l * (l + 1) + m

    def normalize(self) -> "Ylm":
        """Return a new Ylm instance with coefficients normalized to :math:`Y_{0,0}`.

        Returns:
            Ylm instance with normalized coefficients.

        Raises:
            ValueError: if the (0, 0) coefficient is zero.
        """
        data = {k: v / self.data[(0, 0)] for k, v in self.data.items()}
        return Ylm(data=data)

    def tosparse(self) -> BCOO:
        """Return a sparse (jax.experimental.sparse.BCOO) spherical harmonic
        coefficients vector where the spherical harmonic :math:`Y_{l, m}` is located at
        the index :math:`n = l^2 + l + m`.
        """
        indices, values = zip(*self.data.items(), strict=False)
        idx = jnp.array([Ylm.index(l, m) for l, m in indices])[:, None]
        return BCOO((jnp.asarray(values), idx), shape=self.shape)

    def todense(self) -> Array:
        """Return a dense spherical harmonic coefficients vector where the spherical
        harmonic :math:`Y_{l, m}` is located at the index :math:`n = l^2 + l + m`.
        """
        return self.tosparse().todense()

    @classmethod
    def from_dense(cls, y: Array, normalize: bool = True) -> "Ylm":
        """Create a Ylm object from a dense array of spherical harmonic coefficients
        where the spherical harmonic :math:`Y_{l, m}` is located at the index
        :math:`n = l^2 + l + m`.
        """
        data = {}
        for i, ylm in enumerate(y):
            l = int(np.floor(np.sqrt(i)))
            m = i - l * (l + 1)
            data[(l, m)] = ylm
        ylm = cls(data)
        if normalize:
            return ylm.normalize()
        else:
            return ylm

    def __mul__(self, other: Any) -> "Ylm":
        if isinstance(other, Ylm):
            return _mul(self, other)
        else:
            return jax.tree_util.tree_map(lambda x: x * other, self)

    def __rmul__(self, other: Any) -> "Ylm":
        assert not isinstance(other, Ylm)
        return jax.tree_util.tree_map(lambda x: other * x, self)

    def __getitem__(self, key) -> Array:
        assert isinstance(key, tuple)
        return self.todense()[Ylm.index(*key)]

    @classmethod
    def from_limb_darkening(cls, u: Array) -> "Ylm":
        """
        Spherical harmonics coefficients from limb darkening coefficients.
        """
        deg = len(u)
        _u = np.array([1, *u])
        pu = _u @ basis.U(deg)
        yu = np.array(np.linalg.inv(basis.A1(deg).todense()) @ pu)
        yu = Ylm.from_dense(yu.flatten(), normalize=False)
        norm = 1 / (Pijk.from_dense(pu, degree=deg).tosparse() @ solution.rT(deg))
        return yu * norm


def _mul(f: Ylm, g: Ylm) -> Ylm:
    """
    Based closely on the implementation from the MIT-licensed spherical package:

    https://github.com/moble/spherical/blob/0aa81c309cac70b90f8dfb743ce35d2cc9ae6dee/
    spherical/multiplication.py
    """
    ellmax_f = f.deg
    ellmax_g = g.deg
    ellmax_fg = ellmax_f + ellmax_g
    fg = defaultdict(lambda *_: 0.0)
    m_calculator = Wigner3jCalculator(ellmax_f, ellmax_g)
    for ell1 in range(ellmax_f + 1):
        sqrt1 = math.sqrt((2 * ell1 + 1) / (4 * math.pi))
        for m1 in range(-ell1, ell1 + 1):
            idx1 = (ell1, m1)
            if idx1 not in f.data:
                continue
            sum1 = sqrt1 * f.data[idx1]
            for ell2 in range(ellmax_g + 1):
                sqrt2 = math.sqrt(2 * ell2 + 1)
                # w3j_s = s_calculator.calculate(ell1, ell2, s_f, s_g)
                for m2 in range(-ell2, ell2 + 1):
                    idx2 = (ell2, m2)
                    if idx2 not in g.data:
                        continue
                    w3j_m = m_calculator.calculate(ell1, ell2, m1, m2)
                    sum2 = sqrt2 * g.data[idx2]
                    m3 = m1 + m2
                    for ell3 in range(
                        max(abs(m3), abs(ell1 - ell2)), min(ell1 + ell2, ellmax_fg) + 1
                    ):
                        # Could loop over same (ell3, m3) more than once, so add all
                        # contributions together
                        fg[(ell3, m3)] += (
                            (
                                math.pow(-1, ell1 + ell2 + ell3 + m3)
                                * math.sqrt(2 * ell3 + 1)
                                * w3j_m[ell3]  # Wigner3j(ell1, ell2, ell3, m1, m2, -m3)
                            )
                            * sum1
                            * sum2
                        )
    return Ylm(fg)


def Bp(ydeg, npts: int = 1000, eps: float = 1e-9, smoothing=None):
    """
    Return the matrix B+. This expands the
    spot profile `b` in Legendre polynomials. From https://github.com/rodluger/
    mapping_stellar_surfaces/blob/paper2-arxiv/paper2/figures/spot_profile.py and
    _spot_setup in starry/_core/core.py.
    """
    if smoothing is None:
        if ydeg < 4:
            smoothing = 0.5
        else:
            smoothing = 2.0 / ydeg

    theta = jnp.linspace(0, jnp.pi, npts)
    cost = jnp.cos(theta)
    B = jnp.hstack(
        [
            jnp.sqrt(2 * l + 1) * LegendreP(l)(cost).reshape(-1, 1)
            for l in range(ydeg + 1)
        ]
    )
    _Bp = jnp.linalg.solve(B.T @ B + eps * jnp.eye(ydeg + 1), B.T)
    l = jnp.arange(ydeg + 1)
    indices = l * (l + 1)
    S = jnp.exp(-0.5 * indices * smoothing**2)
    return (S[:, None] * _Bp, theta, indices)


def spot_profile(theta, radius, spot_fac=300):
    """
    The sigmoid spot profile.

    """
    z = spot_fac * (theta - radius)
    return 1 / (1 + jnp.exp(-z)) - 1


def ylm_spot(ydeg: int) -> callable:
    """spot expansion in the spherical harmonics basis.

    Args:
        ydeg (int): max degree of the spherical harmonics

    Returns:
        callable: function that returns the spherical harmonics coefficients of the spot
    """
    B, theta, indices = Bp(ydeg)

    def func(contrast: float, r: float, lat: float = 0.0, lon: float = 0.0):
        """spot expansion in the spherical harmonics basis.

        Args:
            contrast (float): spot contrast, defined as (1-c) where c is the intensity
            of the center of the spot relative to the unspotted surface. A contrast of 1.
            means that the spot intensity drops to zero at the center, 0. means that
            the intensity at the center of the spot is the same as the intensity of the
            unspotted surface.
            r (float): radius of the spot.
            lat (float, optional): latitude of the spot, assuming that the center of a
            star with an inclination of pi/2 has a latitude of 0. Defaults to 0.0.
            lon (float, optional): longitude of the spot, assuming that the center of a
            star with an inclination of pi/2 has a longitude of 0. Defaults to 0.0.

        Returns:
            Ylm: Ylm object containing the spherical harmonics coefficients of the spot
        """
        b = spot_profile(theta, r)
        y = jnp.zeros((ydeg + 1) * (ydeg + 1))
        y = y.at[indices].set(B @ b * contrast)
        y = y.at[0].set(y[0] + 1.0)
        y = dot_rotation_matrix(ydeg, 1.0, 0.0, 0.0, lat)(y)
        y = dot_rotation_matrix(ydeg, 0.0, 1.0, 0.0, -lon)(y)

        return Ylm.from_dense(y, normalize=False)

    return func


# TODO
def ring_y(l_max, pts=1000, eps=1e-9, smoothing=None):
    Bp, theta, idxs = None, None, None
    n_max = l_max**2 + 2 * l_max + 1

    def _y(contrast: float, width: float, latitude: float):
        b = 1 - jnp.array((theta > latitude - width) & (theta < latitude + width))
        y = jnp.zeros(n_max)
        y = y.at[idxs].set(Bp @ b)
        return y * contrast

    return _y
