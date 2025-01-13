from collections import defaultdict

from jaxoplanet.starry.core.basis import U
from jaxoplanet.starry.core.polynomials import Pijk
from jaxoplanet.starry.multiprecision import mp, utils
from jaxoplanet.starry.multiprecision.basis import A1, A2
from jaxoplanet.starry.multiprecision.rotation import (
    dot_rotation_matrix,
    dot_rz,
    get_R,
)
from jaxoplanet.starry.multiprecision.solution import get_sT, rT

CACHED_MATRICES = defaultdict(
    lambda: {
        "R_obl": {},
        "R_inc": {},
        "sT": {},
    }
)


def flux(ydeg=0, udeg=0, inc=mp.pi / 2, obl=0.0, cache=None):
    if cache is None:
        cache = CACHED_MATRICES

    deg = ydeg + udeg

    _rT = rT(deg)
    _A2 = A2(deg, cache=cache)

    if ydeg > 0:
        _A1 = A1(ydeg, cache=cache)
        R_px = get_R("R_px", ydeg, cache=cache)
        R_mx = get_R("R_mx", ydeg, cache=cache)
        R_obl = get_R("R_obl", ydeg, obl=obl, inc=inc, cache=cache)
        R_inc = get_R("R_inc", ydeg, obl=obl, inc=inc, cache=cache)
    else:
        _A1 = A1(0, cache=cache)

    if udeg > 0:
        rT_udeg = rT(udeg)

    def rotate_y(y, phi):
        y_rotated = dot_rotation_matrix(ydeg, None, None, R_px)(y)
        y_rotated = dot_rz(ydeg, phi)(y_rotated)
        y_rotated = dot_rotation_matrix(ydeg, None, None, R_mx)(y_rotated)
        y_rotated = dot_rotation_matrix(ydeg, None, None, R_obl)(y_rotated)
        y_rotated = dot_rotation_matrix(ydeg, None, None, R_inc)(y_rotated)
        return y_rotated

    def rot_flux(y, phi):
        y = mp.matrix(y.tolist())
        y_rotated = rotate_y(y, phi, cache=cache)
        return ((_A1 @ y_rotated).T @ _rT)[0]

    def occ_flux(y, b, r, phi=0.0, u=None, rotate=True):
        xo = 0.0
        yo = b
        theta_z = mp.atan2(xo, yo)
        _sT = get_sT(deg, b, r, cache=cache)
        x = _sT.T @ _A2

        if ydeg > 0:
            y = mp.matrix(y.tolist())
            if rotate:
                y_rotated = rotate_y(y, phi)
                y_rotated = dot_rz(ydeg, theta_z)(y_rotated)
        else:
            y_rotated = mp.matrix([1])

        py = (_A1 @ y_rotated).T

        if udeg > 0:
            _U = utils.to_mp(U(udeg))
            u = mp.matrix([1, *u])
            pu = u.T @ _U
            norm = mp.pi / (pu @ rT_udeg)[0]
            py = Pijk.from_dense(pu.tolist()[0]) * Pijk.from_dense(py)
            py = mp.matrix(
                [py.data.get(Pijk.n2ijk(i), 0.0) for i in range((py.degree + 1) ** 2)]
            ).T
        else:
            norm = 1
        return norm * py @ x.T

    def impl(b, r, y=(), phi=0.0, u=(), rotate=True):
        assert len(u) == udeg, "u must have length udeg"
        if ydeg > 0:
            assert len(y) == (ydeg + 1) ** 2, "y must have length (ydeg + 1) ** 2"

        if abs(b) >= (r + 1):
            return rot_flux(y, phi)
        elif r > 1 and abs(b) <= (r - 1):
            return mp.mpf(0.0)
        else:
            return occ_flux(y, b, r, phi, u=u, rotate=rotate)[0]

    return impl
