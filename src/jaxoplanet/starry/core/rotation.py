import math
from functools import partial

import jax
import jax.numpy as jnp

from jaxoplanet.starry.core.s2fft_rotation import (
    compute_rotation_matrices as compute_rotation_matrices_s2fft,
)
from jaxoplanet.types import Array
from jaxoplanet.utils import get_dtype_eps


def dot_rotation_matrix(ydeg, x, y, z, theta):
    """Construct a callable to apply a spherical harmonic rotation

    Args:
        ydeg (int): The order of the spherical harmonic map
        x (float): The x component of the rotation axis
        y (float): The y component of the rotation axis
        z (float): The z component of the rotation axis
        theta (float): The rotation angle in radians
    """
    try:
        ydeg = int(ydeg)
    except TypeError as e:
        raise TypeError(f"ydeg must be an integer; got {ydeg}") from e

    if theta is None:

        def do_dot(M):
            return M

        return do_dot

    if x is None and y is None:
        if z is None:
            raise ValueError("Either x, y, or z must be specified")

        return dot_rz(ydeg, theta)

    x = 0.0 if x is None else x
    y = 0.0 if y is None else y
    z = 0.0 if z is None else z

    if jnp.shape(x) != ():
        raise ValueError(f"x must be a scalar; got {jnp.shape(x)}")
    if jnp.shape(y) != ():
        raise ValueError(f"y must be a scalar; got {jnp.shape(y)}")
    if jnp.shape(z) != ():
        raise ValueError(f"z must be a scalar; got {jnp.shape(z)}")
    if jnp.shape(theta) != ():
        raise ValueError(f"theta must be a scalar; got {jnp.shape(theta)}")

    rotation_matrices = compute_rotation_matrices_s2fft(ydeg, x, y, z, theta)
    n_max = ydeg**2 + 2 * ydeg + 1

    @jax.jit
    @partial(jnp.vectorize, signature=f"({n_max})->({n_max})")
    def do_dot(M):
        """Rotate a spherical harmonic map

        Args:
            M (Array[..., n_max]): The spherical harmonic map to rotate
        """
        if M.shape[-1] != n_max:
            raise ValueError(
                f"Dimension mismatch: Input array must have shape (..., {n_max}); "
                f"got {M.shape}"
            )
        result = []
        for ell in range(ydeg + 1):
            result.append(
                M[ell * ell : ell * ell + 2 * ell + 1] @ rotation_matrices[ell]
            )
        return jnp.concatenate(result, axis=0)

    return do_dot


def full_rotation_axis_angle(
    inc: float | None, obl: float | None, theta: float | None, theta_z: float | None
):
    """Return the axis-angle representation of the full rotation of the
       spherical harmonic map

    Args:
        inc (float): map inclination
        obl (float): map obliquity
        theta (float): rotation angle about the map z-axis
        theta_z (float): rotation angle about the sky y-axis

    Returns:
        tuple: x, y, z, angle
    """
    inc = 0.0 if inc is None else inc
    obl = 0.0 if obl is None else obl
    theta = 0.0 if theta is None else theta
    theta_z = 0.0 if theta_z is None else theta_z

    f = 0.5 * math.sqrt(2)
    si = jnp.sin(inc / 2)
    ci = jnp.cos(inc / 2)

    if theta is not None and theta_z is not None:
        sp = jnp.sin(0.5 * (obl + theta + theta_z))
        cp = jnp.cos(0.5 * (obl + theta + theta_z))
        sm = jnp.sin(0.5 * (obl - theta + theta_z))
        cm = jnp.cos(0.5 * (obl - theta + theta_z))
    else:
        sp = jnp.sin(obl / 2)
        cp = jnp.cos(obl / 2)
        sm = sp
        cm = cp

    numerator1 = f * (-si * cm + ci * cp)
    numerator2 = f * (-si * sm + sp * ci)
    numerator3 = f * (si * sm + sp * ci)
    arg = si * cm + ci * cp
    denominator = jnp.sqrt(1 - 0.5 * arg**2)

    farg = f * arg
    zero_angle = jnp.allclose(farg, 1.0, atol=get_dtype_eps(farg))
    angle = jnp.where(zero_angle, 0.0, 2 * jnp.arccos(farg))

    non_zero_angle = jnp.logical_not(zero_angle)

    # this is mostly useful for the float32 case, where
    # (1 - 0.5 * arg**2) in denominator can be negative due to numerical error
    positive_arg = arg**2 < 2.0
    axis_x = jnp.where(positive_arg & non_zero_angle, numerator1 / denominator, 1.0)
    axis_y = jnp.where(positive_arg & non_zero_angle, numerator2 / denominator, 0.0)
    axis_z = jnp.where(positive_arg & non_zero_angle, numerator3 / denominator, 0.0)

    return axis_x, axis_y, axis_z, angle


def sky_projection_axis_angle(inc: float | None, obl: float | None):
    """Return the axis-angle representation of the partial rotation of the
       map due to inclination and obliquity

    Args:
        inc (float or None): map inclination
        obl (float or None): map obliquity

    Returns:
        tuple: x, y, z, angle
    """

    if obl is None and inc is None:
        return 1.0, None, None, None

    elif obl is None:
        return 1.0, None, None, inc

    elif inc is None:
        return None, None, 1.0, obl

    else:
        co = jnp.cos(obl / 2)
        so = jnp.sin(obl / 2)
        ci = jnp.cos(inc / 2)
        si = jnp.sin(inc / 2)

        denominator = jnp.sqrt(1 - ci**2 * co**2)

        # to avoid nans for the case where ci * co == 1
        denominator = jnp.where(denominator == 0.0, 1.0, denominator)

        axis_x = si * co
        axis_y = si * so
        axis_z = -so * ci

        angle = 2 * jnp.arccos(ci * co)

        arg = jnp.linalg.norm(jnp.array([axis_x, axis_y, axis_z]))
        axis_x = jnp.where(arg > 0, axis_x / denominator, 1.0)
        axis_y = jnp.where(arg > 0, axis_y / denominator, 0.0)
        axis_z = jnp.where(arg > 0, axis_z / denominator, 0.0)

        return axis_x, axis_y, axis_z, angle


def left_project(
    ydeg: int,
    inc: float | None,
    obl: float | None,
    theta: float | None,
    theta_z: float | None,
    y: Array,
):
    """R @ y

    Args:
        ydeg (int): degree of the spherical harmonic map
        inc (float or None): map inclination
        obl (float or None): map obliquity
        theta (float or None): rotation angle about the map z-axis
        theta_z (float or None): rotation angle about the sky y-axis
        x (Array): spherical harmonic map coefficients

    Returns:
        Array: rotated spherical harmonic map coefficients
    """
    m_theta = -theta if theta is not None else theta
    m_theta_z = -theta_z if theta_z is not None else theta_z

    axis_x, axis_y, axis_z, angle = sky_projection_axis_angle(inc, obl)
    y = dot_rotation_matrix(ydeg, 1.0, None, None, -0.5 * jnp.pi)(y)
    y = dot_rotation_matrix(ydeg, None, None, 1.0, m_theta)(y)
    y = dot_rotation_matrix(ydeg, axis_x, axis_y, axis_z, angle)(y)
    y = dot_rotation_matrix(ydeg, None, None, 1.0, m_theta_z)(y)
    return y


def right_project(
    ydeg: int,
    inc: float | None,
    obl: float | None,
    theta: float | None,
    theta_z: float | None,
    y: Array,
):
    """y @ R

    Args:
        ydeg (int): degree of the spherical harmonic map
        inc (float or None): map inclination
        obl (float or None): map obliquity
        theta (float or None): rotation angle about the map z-axis
        theta_z (float or None): rotation angle about the sky y-axis
        x (Array): spherical harmonic map coefficients

    Returns:
        Array: rotated spherical harmonic map coefficients
    """
    axis_x, axis_y, axis_z, angle = sky_projection_axis_angle(inc, obl)
    m_axis_x = -axis_x if axis_x is not None else axis_x
    m_axis_y = -axis_y if axis_y is not None else axis_y
    m_axis_z = -axis_z if axis_z is not None else axis_z

    y = dot_rotation_matrix(ydeg, None, None, 1.0, theta_z)(y)
    y = dot_rotation_matrix(ydeg, m_axis_x, m_axis_y, m_axis_z, angle)(y)
    y = dot_rotation_matrix(ydeg, None, None, 1.0, theta)(y)
    y = dot_rotation_matrix(ydeg, 1.0, None, None, 0.5 * jnp.pi)(y)
    return y


@partial(jax.jit, static_argnums=(0,))
def compute_rotation_matrices(ydeg, x, y, z, theta):
    # we need the axis to be a unit vector - enforce that here
    norm = jnp.sqrt(x * x + y * y + z * z)
    # handle the case where axis is (0, 0, 0)
    x = jnp.where(norm == 0.0, 0.0, x / norm)
    y = jnp.where(norm == 0.0, 0.0, y / norm)
    z = jnp.where(norm == 0.0, 0.0, z / norm)

    s = jnp.sin(theta)
    c = jnp.cos(theta)
    ra01 = x * y * (1 - c) - z * s
    ra02 = x * z * (1 - c) + y * s
    ra11 = c + y * y * (1 - c)
    ra12 = y * z * (1 - c) - x * s
    ra20 = z * x * (1 - c) - y * s
    ra21 = z * y * (1 - c) + x * s
    ra22 = c + z * z * (1 - c)

    tol = 10 * get_dtype_eps(ra22)
    cond_neg = jnp.less(jnp.abs(ra22 + 1.0), tol)
    cond_pos = jnp.less(jnp.abs(ra22 - 1.0), tol)
    cond_full = jnp.logical_or(cond_pos, cond_neg)
    sign = cond_neg.astype(int) - cond_pos.astype(int)
    norm1 = jnp.sqrt(jnp.where(cond_full, 1, ra20 * ra20 + ra21 * ra21))
    norm2 = jnp.sqrt(jnp.where(cond_full, 1, ra02 * ra02 + ra12 * ra12))
    cos_beta = ra22
    ra22_ = jnp.where(cond_full, 0.0, ra22)
    sin_beta = jnp.where(
        cond_full,
        1 + sign * ra22,
        jnp.sqrt(1 - ra22_ * ra22_),  # type: ignore
    )
    cos_gamma = jnp.where(cond_full, ra11, -ra20 / norm1)
    sin_gamma = jnp.where(cond_full, sign * ra01, ra21 / norm1)
    cos_alpha = jnp.where(cond_full, -sign * ra22, ra02 / norm2)
    sin_alpha = jnp.where(cond_full, 1 + sign * ra22, ra12 / norm2)

    return rotar(
        ydeg,
        cos_alpha,
        sin_alpha,
        cos_beta,
        sin_beta,
        cos_gamma,
        sin_gamma,
    )[1]


def rotar(ydeg, c1, s1, c2, s2, c3, s3):
    sqrt_2 = jnp.sqrt(2.0)

    D = []
    R = []

    # D[0]; R[0 ]
    D.append(jnp.ones((1, 1)))
    R.append(jnp.ones((1, 1)))
    if ydeg == 0:
        return D, R

    # D[1]
    D1_22 = 0.5 * (1 + c2)
    D1_21 = -s2 / sqrt_2
    D1_20 = 0.5 * (1 - c2)
    D1_12 = -D1_21
    D1_11 = D1_22 - D1_20
    D1_10 = D1_21
    D1_02 = D1_20
    D1_01 = D1_12
    D1_00 = D1_22
    D.append(
        jnp.array(
            [
                [D1_00, D1_01, D1_02],
                [D1_10, D1_11, D1_12],
                [D1_20, D1_21, D1_22],
            ]
        )
    )

    # R[1]
    cosag = c1 * c3 - s1 * s3
    cosamg = c1 * c3 + s1 * s3
    sinag = s1 * c3 + c1 * s3
    sinamg = s1 * c3 - c1 * s3
    R1_11 = D1_11
    R1_21 = sqrt_2 * D1_12 * c1
    R1_01 = sqrt_2 * D1_12 * s1
    R1_12 = sqrt_2 * D1_21 * c3
    R1_10 = -sqrt_2 * D1_21 * s3
    R1_22 = D1_22 * cosag - D1_20 * cosamg
    R1_20 = -D1_22 * sinag - D1_20 * sinamg
    R1_02 = D1_22 * sinag - D1_20 * sinamg
    R1_00 = D1_22 * cosag + D1_20 * cosamg
    R.append(
        jnp.array(
            [
                [R1_00, R1_01, R1_02],
                [R1_10, R1_11, R1_12],
                [R1_20, R1_21, R1_22],
            ]
        )
    )

    tol = 10 * jnp.finfo(jnp.dtype(s2)).eps
    s2_cond = jnp.less(jnp.abs(s2), tol)
    tgbet2 = jnp.where(s2_cond, s2, (1 - c2) / jnp.where(s2_cond, 1.0, s2))

    for ell in range(2, ydeg + 1):
        D_, R_ = dlmn(ell, s1, c1, c2, tgbet2, s3, c3, D)
        D.append(D_)
        R.append(R_)

    return D, R


def dlmn(ell, s1, c1, c2, tgbet2, s3, c3, D):
    iinf = 1 - ell
    isup = -iinf

    # Last row by recurrence (Eq. 19 and 20 in Alvarez Collado et al.)
    D_ = [[0 for _ in range(2 * ell + 1)] for _ in range(2 * ell + 1)]
    D_[2 * ell][2 * ell] = 0.5 * D[-1][isup + ell - 1, isup + ell - 1] * (1 + c2)
    for m in range(isup, iinf - 1, -1):
        D_[2 * ell][m + ell] = (
            -tgbet2 * jnp.sqrt((ell + m + 1) / (ell - m)) * D_[2 * ell][m + 1 + ell]
        )
    D_[2 * ell][0] = 0.5 * D[ell - 1][isup + ell - 1, -isup + ell - 1] * (1 - c2)

    # The rows of the upper quarter triangle of the D[l;m',m) matrix
    # (Eq. 21 in Alvarez Collado et al.)
    al = ell
    al1 = al - 1
    tal1 = al + al1
    ali = 1.0 / al1
    cosaux = c2 * al * al1
    for mp in range(ell - 1, -1, -1):
        amp = mp
        laux = ell + mp
        lbux = ell - mp
        aux = ali / jnp.sqrt(laux * lbux)
        cux = jnp.sqrt((laux - 1) * (lbux - 1)) * al
        for m in range(isup, iinf - 1, -1):
            am = m
            lauz = ell + m
            lbuz = ell - m
            auz = 1.0 / jnp.sqrt(lauz * lbuz)
            fact = aux * auz
            term = tal1 * (cosaux - am * amp) * D[-1][mp + ell - 1, m + ell - 1]
            if lbuz != 1 and lbux != 1:
                cuz = jnp.sqrt((lauz - 1) * (lbuz - 1))
                term = term - D[-2][mp + ell - 2, m + ell - 2] * cux * cuz
            D_[mp + ell][m + ell] = fact * term
        iinf += 1
        isup -= 1

    # The remaining elements of the D[l;m',m) matrix are calculated
    # using the corresponding symmetry relations:
    # reflection ---> ((-1)**(m-m')) D[l;m,m') = D[l;m',m), m'<=m
    # inversion ---> ((-1)**(m-m')) D[l;-m',-m) = D[l;m',m)
    sign = 1
    iinf = -ell
    isup = ell - 1
    for m in range(ell, 0, -1):
        for mp in range(iinf, isup + 1):
            D_[mp + ell][m + ell] = sign * D_[m + ell][mp + ell]
            sign *= -1
        iinf += 1
        isup -= 1

    # Inversion
    iinf = -ell
    isup = iinf
    for m in range(ell - 1, -(ell + 1), -1):
        sign = -1
        for mp in range(isup, iinf - 1, -1):
            D_[mp + ell][m + ell] = sign * D_[-mp + ell][-m + ell]
            sign *= -1
        # iinf += 1
        isup += 1

    # Compute the real rotation matrices R from the complex ones D
    R_ = [[0 for _ in range(2 * ell + 1)] for _ in range(2 * ell + 1)]
    R_[ell][ell] = D_[ell][ell]
    cosmal = c1
    sinmal = s1
    sign = -1
    root_two = jnp.sqrt(2.0)
    for mp in range(1, ell + 1):
        cosmga = c3
        sinmga = s3
        aux = root_two * D_[ell][mp + ell]
        R_[mp + ell][ell] = aux * cosmal
        R_[-mp + ell][ell] = aux * sinmal
        for m in range(1, ell + 1):
            aux = root_two * D_[m + ell][ell]
            R_[ell][m + ell] = aux * cosmga
            R_[ell][-m + ell] = -aux * sinmga
            d1 = D_[-mp + ell][-m + ell]
            d2 = sign * D_[mp + ell][-m + ell]
            cosag = cosmal * cosmga - sinmal * sinmga
            cosagm = cosmal * cosmga + sinmal * sinmga
            sinag = sinmal * cosmga + cosmal * sinmga
            sinagm = sinmal * cosmga - cosmal * sinmga
            R_[mp + ell][m + ell] = d1 * cosag + d2 * cosagm
            R_[mp + ell][-m + ell] = -d1 * sinag + d2 * sinagm
            R_[-mp + ell][m + ell] = d1 * sinag + d2 * sinagm
            R_[-mp + ell][-m + ell] = d1 * cosag - d2 * cosagm
            aux = cosmga * c3 - sinmga * s3
            sinmga = sinmga * c3 + cosmga * s3
            cosmga = aux
        sign *= -1
        aux = cosmal * c1 - sinmal * s1
        sinmal = sinmal * c1 + cosmal * s1
        cosmal = aux

    return jnp.asarray(D_), jnp.asarray(R_)


def dot_rz(deg, theta):
    """Special case for rotation only around z axis"""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    cosnt = [1.0, c]
    sinnt = [0.0, s]
    for n in range(2, deg + 1):
        cosnt.append(2.0 * cosnt[n - 1] * c - cosnt[n - 2])
        sinnt.append(2.0 * sinnt[n - 1] * c - sinnt[n - 2])

    n = 0
    cosmt = []
    sinmt = []
    for ell in range(deg + 1):
        for m in range(-ell, 0):
            cosmt.append(cosnt[-m])
            sinmt.append(-sinnt[-m])
        for m in range(ell + 1):
            cosmt.append(cosnt[m])
            sinmt.append(sinnt[m])

    n_max = deg**2 + 2 * deg + 1

    @jax.jit
    @partial(jnp.vectorize, signature=f"({n_max})->({n_max})")
    def impl(M):
        result = [0 for _ in range(n_max)]
        for ell in range(deg + 1):
            for j in range(2 * ell + 1):
                result[ell * ell + j] = (
                    M[ell * ell + j] * cosmt[ell * ell + j]
                    + M[ell * ell + 2 * ell - j] * sinmt[ell * ell + j]
                )

        return jnp.array(result, dtype=jnp.dtype(M))

    return impl
