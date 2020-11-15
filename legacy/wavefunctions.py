from math import (
    sqrt, cos, sin, atan2
)

from cmath import exp

import numpy as np

_i = 1j
_isq2 = 1 / sqrt(2)
_A = 1 / sqrt(2 * np.pi)

# these ignore the 1/sqrt(2pi) coefficient
_C100 = _A * sqrt(2)


_C200 = _A * 1 / 4

_C210 = _A * 1 / 4
_C21p1 = _A * 1 / (4 * sqrt(2))
_C21m1 = _C21p1


_C300 = _A * sqrt(2) / (81 * sqrt(3))

_C310 = _A * 2 / 81
_C31p1 = _A * 2 / 81
_C31m1 = _C31p1

_C320 = _A * 1 / (81 * sqrt(3))
_C32p1 = _A * sqrt(2) / 81
_C32m1 = _C32p1
_C32p2 = _A * sqrt(2) / 162
_C32m2 = _C32p2


def sphericals(x, y, z):
    R = x**2 + y**2
    return sqrt(R + z**2), atan2(sqrt(R), z), atan2(y, x)


def cartesians(r, theta, phi):
    st = sin(theta)
    return (
        r * st * cos(phi),
        r * st * sin(phi),
        r * cos(theta)
    )


def rotateabout(nx, ny, nz, incoords="c"):
    """Curried function. Pass a NORMALIZED vector."""

    if incoords == "s":

        k = np.array(cartesians(nx, ny, nz))

        def inner(angle, r, theta, phi):

            v = np.array(cartesians(r, theta, phi))
            c = cos(angle)

            # Rodrigues formula
            return sphericals(
                v * c +
                np.cross(k, v) * sin(angle) +
                k * np.dot(k, v) * (1 - c)
            )

        return inner

    elif incoords == "c":

        k = np.array([nx, ny, nz])

        def inner(angle, x, y, z):

            v = np.array([x, y, z])
            c = cos(angle)

            # Rodrigues formula
            return (
                v * c +
                np.cross(k, v) * sin(angle) +
                k * np.dot(k, v) * (1 - c)
            )

        return inner

    raise ValueError(f"incoords='{incoords}' is invalid")


# n = 1

def psi100(r, theta, phi):
    return _C100 * exp(-r)


# n = 2

def psi200(r, theta, phi):
    return _C200 * (2 - r) * exp(-r / 2)


def psi210(r, theta, phi):
    return _C210 * r * cos(theta) * exp(-r / 2)


def psi21p1(r, theta, phi):
    return _C21p1 * r * sin(theta) * exp(+ _i * phi) * exp(-r / 2)


def psi21m1(r, theta, phi):
    return _C21m1 * r * sin(theta) * exp(- _i * phi) * exp(-r / 2)


# n = 3

# l = 0
def psi300(r, theta, phi):
    return _C300 * (27 - 18 * r + 2 * r**2) * exp(-r / 3)


# l = 1
def psi310(r, theta, phi):
    return _C310 * cos(theta) * (6 - r) * r * exp(-r / 3)


def psi31p1(r, theta, phi):
    return _C31p1 * sin(theta) * exp(+ _i * phi) * (6 - r) * r * exp(-r / 3)


def psi31m1(r, theta, phi):
    return _C31m1 * sin(theta) * exp(- _i * phi) * (6 - r) * r * exp(-r / 3)


# l = 2
def psi320(r, theta, phi):
    return _C320 * (3 * cos(theta)**2 - 1) * r**2 * exp(-r / 3)


def psi32p1(r, theta, phi):
    return _C32p1 * sin(theta) * cos(theta) * exp(+ _i * phi) * r**2 * exp(-r / 3)


def psi32m1(r, theta, phi):
    return _C32p1 * sin(theta) * cos(theta) * exp(- _i * phi) * r**2 * exp(-r / 3)


def psi32p2(r, theta, phi):
    return _C32p2 * sin(theta)**2 * exp(+ _i * 2 * phi) * r**2 * exp(-r / 3)


def psi32m2(r, theta, phi):
    return _C32m2 * sin(theta)**2 * exp(- _i * 2 * phi) * r**2 * exp(-r / 3)


# ORBITAL ALIASES

def orbsp(r, theta, phi):
    return (psi210(r, theta, phi) - psi200(r, theta, phi)) * _isq2


def orb1s(r, theta, phi):
    return psi100(r, theta, phi)


def orb2s(r, theta, phi):
    return psi200(r, theta, phi)


def orb2pz(r, theta, phi):
    return psi210(r, theta, phi)


def orb2px(r, theta, phi):
    return - _isq2 * _i * (psi21p1(r, theta, phi) - psi21m1(r, theta, phi))


def orb2py(r, theta, phi):
    return _isq2 * (psi21p1(r, theta, phi) + psi21m1(r, theta, phi))


def orb3s(r, theta, phi):
    return psi300(r, theta, phi)


def orb3pz(r, theta, phi):
    return psi310(r, theta, phi)


def orb3px(r, theta, phi):
    return (psi31p1(r, theta, phi) - psi31m1(r, theta, phi)) * -_isq2 * _i


def orb3py(r, theta, phi):
    return (psi31p1(r, theta, phi) + psi31m1(r, theta, phi)) * _isq2


def orb3dz2(r, theta, phi):
    return psi320(r, theta, phi)


def orb3dxz(r, theta, phi):
    return (psi32p1(r, theta, phi) - psi32m1(r, theta, phi)) * -_isq2 * _i


def orb3dyz(r, theta, phi):
    return (psi32p1(r, theta, phi) + psi32m1(r, theta, phi)) * _isq2


def orb3dxy(r, theta, phi):
    return (psi32p2(r, theta, phi) - psi32m2(r, theta, phi)) * -_isq2 * _i


def orb3dx2_y2(r, theta, phi):
    return (psi32p2(r, theta, phi) + psi32m2(r, theta, phi)) * -_isq2


def orb2p(nx, ny, nz):
    """Curried function"""

    def orb2pn(r, theta, phi):
        return (
            nx * orb2px(r, theta, phi) +
            ny * orb2py(r, theta, phi) +
            nz * orb2pz(r, theta, phi)
        )

    return orb2pn
