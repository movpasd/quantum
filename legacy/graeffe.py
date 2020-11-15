"""
Implements Graeffe's method for polynomials.
"""

import numpy as np
from numpy import ndarray as NumpyArray


def knuthupperbound(coeffs: NumpyArray) -> float:
    """Get the Knuth upper bound."""

    # Trim and normalise the polynomial.
    coeffs = np.trim_zeros(coeffs, "b")
    coeffs = coeffs / coeffs[-1]

    # Create the set { |a_k-1|, |a_k-2|**1/2, ..., |a0|**1/k }
    powers = np.arange(len(coeffs) - 1, 0, -1)
    powers = 1 / powers
    knuthset = np.abs(coeffs[:-1]) ** powers

    return 2 * max(knuthset)


def pn_add(c1: NumpyArray, c2: NumpyArray) -> NumpyArray:

    s1, s2 = c1.size, c2.size

    if s1 < s2:
        c1 = np.concatenate((c1, np.zeros(s2 - s1)))
    elif s2 < s1:
        c2 = np.concatenate((c2, np.zeros(s1 - s2)))

    return np.trim_zeros(c1 + c2, "b")


def pn_square(coeffs: NumpyArray) -> NumpyArray:
    """Returns square of polynomial."""

    coeffs = np.trim_zeros(coeffs, "b")
    s = coeffs.size
    out = np.zeros(2 * s - 1)
    for k in range(s):
        out[k:k + s] += coeffs[k] * coeffs

    return out


def pn_shift(coeffs: NumpyArray, amount: int) -> NumpyArray:

    if amount == 0:
        return coeffs

    out = np.zeros(coeffs.size + amount)

    if amount > 0:
        out[amount:] = coeffs
    elif amount < 0:
        out = coeffs[-amount:]

    return out


def graeffenext(coeffs: NumpyArray) -> NumpyArray:
    """Returns the next polynomial per Graeffe's method."""

    f = coeffs
    g = pn_square(f[0::2])
    h = pn_square(f[1::2])

    return pn_add(g, -pn_shift(h, +1))


def upperbound(coeffs: NumpyArray, iterations=3) -> float:
    """
    Apply Graeffe's method to polynomial.

    Takes coefficients of the polynomial as a numpy array.
    Returns the upper bound on the modulus of the roots.

    Based off of JH Davenport, M Mignotte,
        "On finding the largest root of a polynomial",
        Modélisation mathématique et analyse numérique,
        tome 24, no6 (1990), p693-696.
    """

    for i in range(iterations):
        coeffs = graeffenext(coeffs)

    return knuthupperbound(coeffs)**(2**-iterations)
