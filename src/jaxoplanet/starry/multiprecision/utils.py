import mpmath
import numpy as np

from jaxoplanet.starry.multiprecision import mp


def to_numpy(x):
    array = np.array(x.tolist(), dtype=np.float64)
    if array.shape[1] == (1):
        return array.T[0]
    else:
        return array


def to_mp(x):
    return mp.matrix(x.tolist())


def diff_mp(M1: np.ndarray | mpmath.matrix, M2: np.ndarray | mpmath.matrix):
    """Returns M1 - M2, at arbitrary precision and casted to numpy.float64.

    This function allows comparison of matrices at arbitrary precision if at least one is
    a mpmath matrix.

    Args:
        M1 (np.ndarray or mpmath.matrix): matrix
        M2 (np.ndarray or mpmath.matrix): matrix

    Returns:
        np.ndarray:
            difference between M1 and M2, casted to numpy.float64
    """
    if isinstance(M1, np.ndarray) and isinstance(M2, np.ndarray):
        d = M1 - M2
    else:
        _M1 = to_mp(M1) if isinstance(M1, np.ndarray) else M1
        _M2 = to_mp(M2) if isinstance(M2, np.ndarray) else M2
        d = to_numpy(_M1 - _M2).astype(np.float64)
    return d


def kron_delta(m, n):
    return 1 if m == n else 0


def fac(n):
    if n == 0:
        return mp.mpf(1.0)
    else:
        try:
            return mp.fac(n)
        except ValueError:
            return mp.inf
