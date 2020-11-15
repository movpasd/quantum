from wavefunctions import *


import numpy as np

NUM = 75
MAX = 100
DX = 2 * MAX / (NUM - 1)

xs, ys, zs = np.meshgrid(
    np.linspace(-MAX, MAX, NUM),
    np.linspace(-MAX, MAX, NUM),
    np.linspace(-MAX, MAX, NUM)
)


def integ1(x, y, z):
    sp = sphericals(x, y, z)
    return np.multiply(
        np.conj(orb1s(*sp)), orb2px(*sp)
    )


def integ2(x, y, z):
    sp = sphericals(x, y, z)
    return np.multiply(
        np.conj(orb2s(*sp)), orb2px(*sp)
    )


du3 = np.vectorize(integ1)(xs, ys, zs)
dv3 = np.vectorize(integ2)(xs, ys, zs)

integral1 = np.sum(du3)
integral2 = np.sum(dv3)

print(integral1 / (2 * np.pi))
print(integral2 / (2 * np.pi))
