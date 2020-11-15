
from typing import Dict, Tuple, Callable, Iterable
from math import exp as floatexp
from math import cos, sin, sqrt, factorial
from cmath import exp
from numpy import pi
from numpy import ndarray as NumpyArray
import numpy as np


OVERALLCONST = 1


# key: n, l
_cache_normconst: Dict[Tuple[int, int], float] = {}

# key: subscript, superscript
#      n-l-1      2l+1
_cache_laguerre: Dict[Tuple[int, int], NumpyArray] = {}

# key: subscript, superscript
#      l          m
_cache_legendre: Dict[Tuple[int, int], NumpyArray] = {}
_cache_harm_norm: Dict[Tuple[int, int], float] = {}

# Note: for odd values of m, the coefficient values are of the
# associated Legendre function divided by sqrt(1-x**2), which is
# a polynomial.


def generate(maxn):
    """
    Prepares polynomials and normalisation constants.

    Generates Laguerre polynomial coefficients up to:
        (n-l-1, 2l+1)
           .. Legendre polynomial coefficients up to:
        (l, m)
    Generates normalisation constants:
        Overall up to n, l
        Harmonics up to l, m
    """

    if maxn > 85:
        raise ValueError(f"maxn={maxn} too large! (>85)")

    maxl = maxn - 1

    # OVERALL NORMALIZATION

    for n in range(maxn + 1):
        for l in range(n):
            _cache_normconst[(n, l)] = sqrt(
                4 / n**4 *
                factorial(n - l - 1) / factorial(n + l)
            ) * OVERALLCONST

    # HARMONIC NORMALIZATION

    for l in range(maxl + 1):
        frontconst = (2 * l + 1) / (4 * pi)
        # calculate m<= and m>=0 separately
        for m in range(-l, l + 1):
            _cache_harm_norm[(l, m)] = sqrt(
                frontconst *
                factorial(l - m) / factorial(l + m)
            )

    # LAGUERRE COEFFICIENTS

    for sup in range(2 * maxl + 1 + 1):

        _cache_laguerre[(0, sup)] = np.array((1,))
        _cache_laguerre[(1, sup)] = np.array((1 + sup, -1))

        for sub in range(1, maxn):
            _cache_laguerre[(sub + 1, sup)] = _laguerre_next(
                sub, sup,
                _cache_laguerre[(sub, sup)],
                _cache_laguerre[(sub - 1, sup)]

            )

    # LEGENDRE COEFFICIENTS

    # Generate all the basic Legendre polynomials (m = 0)

    _cache_legendre[(0, 0)] = np.array((1,))

    _cache_legendre[(1, 0)] = np.array((0, 1))
    _cache_legendre[(1, 1)] = np.array((-1,))
    _cache_legendre[(1, -1)] = np.array((0.5,))

    for l in range(1, maxl):

        _cache_legendre[(l + 1, 0)] = _legendre_l_next(
            l, _cache_legendre[(l, 0)], _cache_legendre[(l - 1, 0)]
        )

    # For each l generate the positive polynomials (m > 0) using
    # the derivatives formula. Then use reflection to get m < 0.

    for l in range(0, maxl + 1):

        # mth derivative
        dv = _diff(_cache_legendre[(l, 0)])

        for m in range(1, l + 1):

            nextleg = dv.copy()

            for q in range(m // 2):
                # Multiply by (1-x**2)**m//2
                nextleg = nextleg - np.roll(nextleg, 2)

            # Condon-Shortley phase
            if m % 2 == 0:
                nextleg = -nextleg

            reflection = factorial(l - m) / factorial(l + m)  # * (-1)**m
            _cache_legendre[(l, m)] = nextleg
            _cache_legendre[(l, -m)] = reflection * nextleg

            dv = _diff(dv)


def _diff(poly: NumpyArray, truncate=False):

    if truncate:
        out = np.zeros(len(poly) - 1)
        out[:] = poly[1:]
        out *= np.arange(1, len(out) + 1)
        return out
    else:
        out = np.zeros(len(poly))
        out[:-1] = poly[1:]
        out *= np.arange(1, len(out) + 1)
        return out


def _laguerre_next(sub: int, sup: int,
                   current: NumpyArray,
                   previous: NumpyArray) -> NumpyArray:

    out: NumpyArray = np.zeros(sub + 2)
    cur: NumpyArray = current.copy()
    prev: NumpyArray = previous.copy()

    out[1:] = -cur
    out[:-1] += (2 * sub + 1 + sup) * cur
    out[:-2] -= (sub + sup) * prev
    out /= sub + 1

    return out


def _legendre_l_next(sub: int,
                     current: NumpyArray,
                     previous: NumpyArray) -> NumpyArray:

    out: NumpyArray = np.zeros(sub + 2)
    cur: NumpyArray = current.copy()
    prev: NumpyArray = previous.copy()

    out[1:] += (2 * sub + 1) * cur
    out[:-2] -= sub * prev
    out /= sub + 1

    return out


def getpsi(n: int, l: int, m: int,
           a0: float = 1, normalised: bool = False):

    try:

        assert (isinstance(n, int) and
                isinstance(l, int) and
                isinstance(m, int))
        assert (n > 0 and
                0 <= l < n and
                -l <= m <= +l)

    except AssertionError:

        print(f"Bad argument nlm {(n, l, m)}")
        raise

    if normalised:

        def psi(r, theta, phi):
            return (
                normconst(n, l, m) *
                getradial(n, l, m)(r / a0) *
                getharmonic(l, m, normalised)(theta, phi)
            )

    else:

        def psi(r, theta, phi):
            return (
                getradial(n, l, m)(r / a0) *
                getharmonic(l, m, normalised)(theta, phi)
            )

    return psi


def normconst(n: int, l: int, m: int) -> float:

    try:
        return _cache_normconst[(n, l)]
    except KeyError:
        print(f"Haven't generated overall normalisation for {(n, l, m)}")
        raise


def getradial(n: int, l: int, m: int) -> Callable[[float], float]:

    try:
        Lcoeffs = _cache_laguerre[(n - l - 1, 2 * l + 1)]
    except KeyError:
        print(f"Haven't generated Laguerre coeffs for nlm{(n, l, m)}")
        raise

    def radial(r: float) -> float:

        rho = 2 * r / n

        Lterms = np.repeat(1., len(Lcoeffs))
        for k in range(1, len(Lcoeffs)):
            Lterms[k:] *= rho
        Lterms *= Lcoeffs

        return (
            floatexp(- rho / 2) *
            rho**l *
            np.sum(Lterms)
        )

    return radial


def getradialderivative(n: int, l: int, m: int) -> Callable[[float], float]:

    try:
        Lcoeffs = _cache_laguerre[(n - l - 1, 2 * l + 1)]
    except KeyError:
        print(f"Haven't generated Laguerre coeffs for nlm{(n, l, m)}")
        raise

    DLcoeffs = _diff(Lcoeffs, truncate=True)

    def radialderivative(r: float) -> float:

        rho = 2 * r / n
        return (
            floatexp(- rho / 2) * (
                (
                    (-0.5 * rho**l + l * rho**(l - 1)) *
                    sum(Lcoeffs[k] * rho**k for k in range(len(Lcoeffs)))
                ) + (
                    rho**l *
                    sum(DLcoeffs[k] * rho**k for k in range(len(DLcoeffs)))
                )
            )
        )

    return radialderivative


def getharmonic(l: int, m: int, normalised: bool = False) -> (
        Callable[[float, float], complex]):

    try:
        Pcoeffs = _cache_legendre[l, m]
    except KeyError:
        print(f"Haven't generated Legendre coeffs for lm{(l, m)}")
        raise

    if normalised:

        try:
            norm = _cache_harm_norm[l, m]
        except KeyError:
            print(f"Haven't generated Legendre normalisation for lm{(l, m)}")

        if m % 2 == 0:

            def harmonic(theta: float, phi: float) -> complex:
                c = cos(theta)
                return (
                    norm *
                    exp(1j * m * phi) *
                    sum(Pcoeffs[k] * c**k for k in range(len(Pcoeffs)))
                )

        elif m % 2 == 1:

            def harmonic(theta: float, phi: float) -> complex:
                c = cos(theta)
                s = sin(theta)
                return (
                    norm *
                    exp(1j * m * phi) *
                    sum(Pcoeffs[k] * c**k for k in range(len(Pcoeffs))) *
                    s
                )

    else:

        if m % 2 == 0:

            def harmonic(theta: float, phi: float) -> complex:
                c = cos(theta)
                return (
                    exp(1j * m * phi) *
                    sum(Pcoeffs[k] * c**k for k in range(len(Pcoeffs)))
                )

        elif m % 2 == 1:

            def harmonic(theta: float, phi: float) -> complex:
                c = cos(theta)
                s = sin(theta)
                return (
                    exp(1j * m * phi) *
                    sum(Pcoeffs[k] * c**k for k in range(len(Pcoeffs))) *
                    s
                )

    return harmonic


def _test(maxn, *args):

    def title(s):
        print("=" * len(s))
        print(s)
        print("=" * len(s))

    generate(maxn)

    if "normconst" in args:

        title("_cache_normconst")

        for n in range(maxn + 1):
            print(f"n={n}")
            for l in range(n):
                print(f"\tl={l}:\t {round(_cache_normconst[(n, l)], 5)}")
            print()

    if "harmnorm" in args:

        title("_cache_harm_norm")

        for l in range(maxn):
            print(f"l={l}")
            for m in range(-l, l + 1):
                print(f"\tm={m}:\t {round(_cache_harm_norm[(l, m)], 5)}")
            print()

    if "laguerre" in args:

        title("_cache_laguerre")

        for l in range(maxn):

            sup = 2 * l + 1
            print(f"sup={sup}")

            for sub in range(maxn):
                print(
                    f"\tsub={sub}:\t {np.around(_cache_laguerre[(sub, sup)], 5)}"
                )
            print()

    if "legendre" in args:

        title("_cache_legendre")

        for l in range(maxn):

            print(f"l={l}")
            for m in range(-l, l + 1):
                print(f"\tm={m}:\t {np.around(_cache_legendre[(l, m)], 5)}")

            print()

    if "harmonicstest" in args:

        title("harmonicstest")

        from random import random

        for l in range(6):
            for m in range(l):
                harm = getharmonic(l, m, normalised=True)
                SAMPLES = 100000
                randvalues = []
                for i in range(SAMPLES):
                    randvalues.append(harm(random() * pi, random() * 2 * pi))

                print(round(abs(np.mean(randvalues)) /
                            _cache_harm_norm[(l, m)], 4) * 100, "%")

                approxintegral = round(np.sum(
                    np.abs(np.array(randvalues))**2 * 4 * pi / SAMPLES
                ), 3)

                print(approxintegral)
                print()
            print("---")
            print()

    if "getradialderivative" in args:

        import matplotlib.pyplot as plt

        title("getradialderivatives")

        xs = np.linspace(0, 100, 1000)
        y1s = np.vectorize(getradial(5, 2, 1))(xs)
        y2s = np.vectorize(getradialderivative(5, 2, 1))(xs)

        plt.plot(xs, y1s, color="black")
        plt.plot(xs, y2s, color="red")
        plt.show()

    if "large_r" in args:

        import matplotlib.pyplot as plt

        title("large_r")

        N = 50
        L = 10
        M = 0

        xs = np.arange(0, 5 * N * (N - 1), 1)
        radialf = np.vectorize(getradial(N, L, M))(xs)

        lm = abs(radialf).max()
        radialf /= lm
        radialf = np.cbrt(radialf)

        plt.plot(xs, radialf, "r")
        rmax1 = N * (N - 1)
        print(f"rmax1 {rmax1}")
        plt.plot([rmax1, rmax1], [-1, 1], "k--")
        plt.plot([1.5 * rmax1, 1.5 * rmax1], [-1, 1], "k--")
        plt.plot([300 + 1.5 * rmax1, 300 + 1.5 * rmax1], [-1, 1], "k--")
        plt.plot([0, 5 * N * (N - 1)], [0, 0])
        plt.show()


if __name__ == "__main__":

    _test(80, "large_r")
