"""Some of the code in this submodule is taken directly from the MIT-licensed
's2fft' package. Below is the original license text:

Copyright (c) 2022 Authors & Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


def _compute_full(dl: jnp.ndarray, beta: float, L: int, el: int) -> jnp.ndarray:
    """from s2fft.recursions.risbo_jax"""

    if el == 0:
        dl = dl.at[el + L - 1, el + L - 1].set(1.0)
        return dl
    if el == 1:
        cosb = jnp.cos(beta)
        sinb = jnp.sin(beta)

        coshb = jnp.cos(beta / 2.0)
        sinhb = jnp.sin(beta / 2.0)
        sqrt2 = jnp.sqrt(2.0)

        dl = dl.at[L - 2, L - 2].set(coshb**2)
        dl = dl.at[L - 2, L - 1].set(sinb / sqrt2)
        dl = dl.at[L - 2, L].set(sinhb**2)

        dl = dl.at[L - 1, L - 2].set(-sinb / sqrt2)
        dl = dl.at[L - 1, L - 1].set(cosb)
        dl = dl.at[L - 1, L].set(sinb / sqrt2)

        dl = dl.at[L, L - 2].set(sinhb**2)
        dl = dl.at[L, L - 1].set(-sinb / sqrt2)
        dl = dl.at[L, L].set(coshb**2)
        return dl
    else:
        coshb = -jnp.cos(beta / 2.0)
        sinhb = jnp.sin(beta / 2.0)
        dd = jnp.zeros((2 * el + 2, 2 * el + 2))

        # First pass
        j = 2 * el - 1
        i = jnp.arange(j)
        k = jnp.arange(j)

        sqrt_jmk = jnp.sqrt(j - k)
        sqrt_kp1 = jnp.sqrt(k + 1)
        sqrt_jmi = jnp.sqrt(j - i)
        sqrt_ip1 = jnp.sqrt(i + 1)

        dlj = dl[k - (el - 1) + L - 1][:, i - (el - 1) + L - 1]

        dd = dd.at[:j, :j].add(
            jnp.einsum("i,k->ki", sqrt_jmi, sqrt_jmk, optimize=True) * dlj * coshb
        )
        dd = dd.at[:j, 1 : j + 1].add(
            jnp.einsum("i,k->ki", -sqrt_ip1, sqrt_jmk, optimize=True) * dlj * sinhb
        )
        dd = dd.at[1 : j + 1, :j].add(
            jnp.einsum("i,k->ki", sqrt_jmi, sqrt_kp1, optimize=True) * dlj * sinhb
        )
        dd = dd.at[1 : j + 1, 1 : j + 1].add(
            jnp.einsum("i,k->ki", sqrt_ip1, sqrt_kp1, optimize=True) * dlj * coshb
        )

        dl = dl.at[-el + L - 1 : el + 1 + L - 1, -el + L - 1 : el + 1 + L - 1].multiply(
            0.0
        )

        j = 2 * el
        i = jnp.arange(j)
        k = jnp.arange(j)

        # Second pass
        sqrt_jmk = jnp.sqrt(j - k)
        sqrt_kp1 = jnp.sqrt(k + 1)
        sqrt_jmi = jnp.sqrt(j - i)
        sqrt_ip1 = jnp.sqrt(i + 1)

        dl = dl.at[-el + L - 1 : el + L - 1, -el + L - 1 : el + L - 1].add(
            jnp.einsum("i,k->ki", sqrt_jmi, sqrt_jmk, optimize=True)
            * dd[:j, :j]
            * coshb,
        )
        dl = dl.at[-el + L - 1 : el + L - 1, L - el : L + el].add(
            jnp.einsum("i,k->ki", -sqrt_ip1, sqrt_jmk, optimize=True)
            * dd[:j, :j]
            * sinhb,
        )
        dl = dl.at[L - el : L + el, -el + L - 1 : el + L - 1].add(
            jnp.einsum("i,k->ki", sqrt_jmi, sqrt_kp1, optimize=True)
            * dd[:j, :j]
            * sinhb,
        )
        dl = dl.at[L - el : L + el, L - el : L + el].add(
            jnp.einsum("i,k->ki", sqrt_ip1, sqrt_kp1, optimize=True)
            * dd[:j, :j]
            * coshb,
        )
        return dl / ((2 * el) * (2 * el - 1))


def _generate_rotate_dls(L: int, beta: float) -> jnp.ndarray:
    """*from s2fft.utils*"""
    dls = []
    dl_iter = jnp.zeros((2 * L - 1, 2 * L - 1)).astype(float)
    for el in range(L):
        dl_iter = _compute_full(dl_iter, beta, L, el)
        dls.append(dl_iter)
    return jnp.stack(dls, axis=0)


def compute_rotation_matrices(deg, x, y, z, theta, homogeneous=False):

    def kron(m, n):
        """Kronecker delta. Can we do better?"""
        return m == n

    def Umn(m, n):
        """Compute the (m, n) term of the transformation
        matrix from complex to real Ylms. This part is from starry"""
        if n < 0:
            term1 = 1j
        elif n == 0:
            term1 = jnp.sqrt(2) / 2
        else:
            term1 = 1
        if (m > 0) and (n < 0) and (n % 2 == 0):
            term2 = -1
        elif (m > 0) and (n > 0) and (n % 2 != 0):
            term2 = -1
        else:
            term2 = 1
        return term1 * term2 * 1 / jnp.sqrt(2) * (kron(m, n) + kron(m, -n) + 0j)

    def U(l):
        """Compute the complete U transformation matrix.. This part is from starry"""
        res = jnp.zeros((2 * l + 1, 2 * l + 1)) + 0j
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                res = res.at[m + l, n + l].set(Umn(m, n))
        return res

    def euler(x, y, z, theta):
        """axis-angle to euler angles"""
        # the jnp where for theta == 0 is to avoid nans when computing grad
        axis = jnp.array([jnp.where(theta == 0.0, 1.0, x), y, z])
        _theta = jnp.where(theta == 0.0, 1.0, theta)
        axis = axis / jnp.linalg.norm(axis)
        r = Rotation.from_rotvec(axis * _theta)
        return jnp.where(theta == 0.0, jnp.array([0.0, 0.0, 0.0]), r.as_euler("zyz"))

    alpha, beta, gamma = euler(x, y, z, theta)
    dls = _generate_rotate_dls(deg + 1, beta)
    N = jnp.arange(-deg, deg + 1)
    m, mp = jnp.meshgrid(N, N)
    Dlm = jnp.exp(-1j * (m * alpha + mp * gamma)) * dls

    u = U(deg)
    u_inv = jnp.conj(u.T)

    # The output of generate_rotate_dls has shape (L, 2L+1, 2L+1)
    Rs_full = jnp.real(u_inv[None, :] @ Dlm @ u[None, :])

    if homogeneous:
        Rs = Rs_full
    else:
        # reformat to a list of matrices with different shape
        # (to match current implementation of rotation.dot_rotation_matrix)
        Rs = []

        for i in range(deg + 1):
            Rs.append(Rs_full[i, deg - i : deg + i + 1, deg - i : deg + i + 1])

    return Rs
