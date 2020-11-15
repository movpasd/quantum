
from numba import njit
import numpy as np
from . import polynomials as pl


cache = {}  # type: ignore


def generate(maxl):

    for l in range(maxl + 1):
        a = asc_coeffs(l)
        for m in range(-l, l + 1):
            cache[(l, m)] = a[l + m]


def get(l, m):

    if (l, m) in cache:
        return cache[(l, m)]
    else:
        a = asc_coeffs(l)
        for q in range(-l, l + 1):
            cache[(l, q)] = a[l + q]
        return cache[(l, m)]


@njit
def asc_coeffs(l):

    factor = np.array([1., 0., -1])  # (1 - x2)

    ret = []
    for k in range(2 * l + 1):
        ret.append(np.array([0.]))

    ret[l] = calculate(l)

    dv = pl.diff(ret[l])
    for m in range(1, l + 1):
        coeffs = np.copy(dv)
        refl = 1

        for q in range(m // 2):
            coeffs = pl.multiply(coeffs, factor)

        if m % 2 == 1:
            refl = -refl

        if l < 10:
            for q in range(-m + 1, m + 1):
                refl *= l + q

        ret[l + m] = coeffs
        ret[l - m] = coeffs / refl
        dv = pl.diff(dv)

    return ret


@njit
def calculate(l):

    # Rodrigues' formula

    ret = pl.pow(np.array([-1., 0., 1.]), l)

    for k in range(l):
        ret = 0.5 * pl.diff(ret) / (k + 1)

    return ret
