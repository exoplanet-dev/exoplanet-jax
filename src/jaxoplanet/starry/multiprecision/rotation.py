from collections import defaultdict

from jaxoplanet.starry.multiprecision import mp
from jaxoplanet.starry.multiprecision.utils import fac, kron_delta

CACHED_MATRICES = defaultdict(
    lambda: {
        "R_obl": {},
        "R_inc": {},
    }
)


def get_R(name, l_max, obl=None, inc=None, cache=None):
    if cache is None:
        cache = CACHED_MATRICES

    if obl is not None and inc is not None:
        if name == "R_obl":
            if obl not in cache[l_max][name]:
                print(f"pre-computing {name}...")
                cache[l_max][name][obl] = R(l_max, (0.0, 0.0, 1.0), -obl)
            return cache[l_max][name][obl]
        elif name == "R_inc":
            if (inc, obl) not in cache[l_max][name]:
                print(f"pre-computing {name}...")
                cache[l_max][name][(inc, obl)] = R(
                    l_max, (-mp.cos(obl), -mp.sin(obl), 0.0), (0.5 * mp.pi - inc)
                )
            return cache[l_max][name][(inc, obl)]
    else:
        if name not in cache[l_max]:
            if name == "R_px":
                print(f"pre-computing {name}...")
                cache[l_max][name] = R(l_max, (1.0, 0.0, 0.0), -0.5 * mp.pi)
            elif name == "R_mx":
                print(f"pre-computing {name}...")
                cache[l_max][name] = R(l_max, (1.0, 0.0, 0.0), 0.5 * mp.pi)

        return cache[l_max][name]


def R(lmax, u, theta):

    def Dmn(l, m, n, alpha, beta, gamma):
        """Compute the (m, n) term of the Wigner D matrix."""
        sumterm = 0
        # Expression diverges when beta = 0
        if beta == 0:
            beta = 10 ** (-mp.dps)
        for k in range(l + m + 1):
            sumterm += (
                (-1) ** k
                * mp.cos(beta / 2) ** (2 * l + m - n - 2 * k)
                * mp.sin(beta / 2) ** (-m + n + 2 * k)
                / (fac(k) * fac(l + m - k) * fac(l - n - k) * fac(n - m + k))
            )

        dmn = (
            sumterm
            * mp.exp(-mp.j * (alpha * n + gamma * m))
            * (-1) ** (n + m)
            * mp.sqrt(fac(l - m) * fac(l + m) * fac(l - n) * fac(l + n))
        )

        return dmn

    def D(l, alpha, beta, gamma):
        res = mp.zeros(2 * l + 1, 2 * l + 1)
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                res[m + l, n + l] = Dmn(l, n, m, alpha, beta, gamma)
        return res

    def Umn(l, m, n):
        """Compute the (m, n) term of the transformation
        matrix from complex to real Ylms."""
        if n < 0:
            term1 = mp.j
        elif n == 0:
            term1 = mp.sqrt(2) / 2
        else:
            term1 = 1
        if (m > 0) and (n < 0) and (n % 2 == 0):
            term2 = -1
        elif (m > 0) and (n > 0) and (n % 2 != 0):
            term2 = -1
        else:
            term2 = 1
        return term1 * term2 * 1 / mp.sqrt(2) * (kron_delta(m, n) + kron_delta(m, -n))

    def U(l):
        """Compute the U transformation matrix."""
        res = mp.zeros(2 * l + 1, 2 * l + 1)
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                res[m + l, n + l] = Umn(l, m, n)
        return res

    def rot_matrix(l, alpha, beta, gamma):
        """Return the rotation matrix for a single degree `l`."""
        res = mp.zeros(2 * l + 1, 2 * l + 1)
        if l == 0:
            res[0, 0] = 1
            return res
        foo = ((U(l) ** -1) * D(l, alpha, beta, gamma) * U(l)).apply(mp.re)
        for m in range(2 * l + 1):
            for n in range(2 * l + 1):
                if abs(foo[m, n]) < 10 ** (-mp.dps):
                    res[m, n] = 0
                else:
                    res[m, n] = foo[m, n]
        return res

    def axis_angle_to_euler(u1, u2, u3, theta):
        """Axis-angle rotation matrix."""

        tol = 1e-20
        if theta == 0:
            theta = tol
        if u1 == 0 and u2 == 0:
            u1 = tol
            u2 = tol

        # Elements of the transformation matrix
        costheta = mp.cos(theta)
        sintheta = mp.sin(theta)
        RA01 = u1 * u2 * (1 - costheta) - u3 * sintheta
        RA02 = u1 * u3 * (1 - costheta) + u2 * sintheta
        RA11 = costheta + u2 * u2 * (1 - costheta)
        RA12 = u2 * u3 * (1 - costheta) - u1 * sintheta
        RA20 = u3 * u1 * (1 - costheta) - u2 * sintheta
        RA21 = u3 * u2 * (1 - costheta) + u1 * sintheta
        RA22 = costheta + u3 * u3 * (1 - costheta)

        # Determine the Euler angles
        if (RA22 < -1) and (RA22 > -1):
            cosbeta = -1
            sinbeta = 0
            cosgamma = RA11
            singamma = RA01
            cosalpha = 1
            sinalpha = 0
        elif (RA22 < 1) and (RA22 > 1):
            cosbeta = 1
            sinbeta = 0
            cosgamma = RA11
            singamma = -RA01
            cosalpha = 1
            sinalpha = 0
        else:
            cosbeta = RA22
            sinbeta = mp.sqrt(1 - cosbeta**2)
            norm1 = mp.sqrt(RA20 * RA20 + RA21 * RA21)
            norm2 = mp.sqrt(RA02 * RA02 + RA12 * RA12)
            cosgamma = -RA20 / norm1
            singamma = RA21 / norm1
            cosalpha = RA02 / norm2
            sinalpha = RA12 / norm2
        alpha = mp.atan2(sinalpha, cosalpha)
        beta = mp.atan2(sinbeta, cosbeta)
        gamma = mp.atan2(singamma, cosgamma)

        return alpha, beta, gamma

    u = mp.matrix(u)
    u = u / mp.norm(u, p=2)
    alpha, beta, gamma = axis_angle_to_euler(*u, theta)
    blocks = [rot_matrix(l, alpha, beta, gamma) for l in range(lmax + 1)]

    return blocks


def dot_rotation_matrix(ydeg, u, theta, rotation_matrices=None):
    if rotation_matrices is None:
        rotation_matrices = R(ydeg, u, theta)

    def do_dot(M):
        if theta is not None:
            if mp.absmax(theta) == 0:
                return M
        result = []
        for l in range(ydeg + 1):
            if l == 0:
                result.append(mp.matrix([1]))
            else:
                result.append(
                    mp.matrix(M[l * l : l * l + 2 * l + 1]).T @ rotation_matrices[l]
                )
        S = []
        for _s in result:
            for __s in _s:
                S.append(__s)

        return mp.matrix(S)

    return do_dot


def left_project(deg, inc, obl, theta, theta_z, x):
    if theta != 0:
        x = dot_rotation_matrix(deg, (1.0, 0.0, 0.0), -0.5 * mp.pi)(x)
        x = dot_rotation_matrix(deg, (0.0, 0.0, 1.0), -theta)(x)
        x = dot_rotation_matrix(deg, (1.0, 0.0, 0.0), 0.5 * mp.pi)(x)

    if obl != 0:
        x = dot_rotation_matrix(deg, (0.0, 0.0, 1.0), -obl)(x)

    x = dot_rotation_matrix(
        deg, (-mp.cos(obl), -mp.sin(obl), 0.0), (0.5 * mp.pi - inc)
    )(x)

    if theta_z != 0:
        x = dot_rotation_matrix(deg, 0, 0, 1.0, -theta_z)(x)
    return x.T


def dot_rz(deg, theta):
    """Special case for rotation only around z axis"""
    c = mp.cos(-theta)
    s = mp.sin(-theta)
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

    def impl(M):
        result = [0 for _ in range(n_max)]
        for ell in range(deg + 1):
            for j in range(2 * ell + 1):
                result[ell * ell + j] = (
                    M[ell * ell + j] * cosmt[ell * ell + j]
                    + M[ell * ell + 2 * ell - j] * sinmt[ell * ell + j]
                )

        return mp.matrix(result)

    return impl
