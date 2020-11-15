"""
Numba-based polynomials.
"""

from numba import njit
import numpy as np


# TODO
#   product() could be optimised


@njit
def eval(coeffs, x):

    length = coeffs.shape[0]
    ret = 0
    for i in range(length):
        ret *= x
        ret += coeffs[length - i - 1]
    return ret


@njit
def add(a, b, trim=True):

    if trim:
        a, b = trimmed(a), trimmed(b)

    asize, bsize = a.shape[0], b.shape[0]

    if asize >= bsize:
        return a + fitsize(b, asize)
    else:
        return b + fitsize(a, bsize)


@njit
def multiply(a, b, trim=True):

    if trim:
        a, b = trimmed(a), trimmed(b)

    asize = a.shape[0]
    bsize = b.shape[0]
    rsize = asize + bsize - 1
    ret = np.zeros(rsize)

    for i in range(asize):
        for j in range(bsize):
            ret[i + j] += a[i] * b[j]

    return ret


@njit
def sum(*args):
    return np.sum(align(*args), 0)


@njit
def product(*args):

    argnum = len(args)

    if argnum == 1:
        return args[0]

    ret = multiply(args[0], args[1])
    for i in range(2, argnum):
        ret = multiply(ret, args[i])

    return ret


@njit
def pow(a, n):
    # Could be optimised a lot

    if n == 0:
        return np.array([1.])

    ret = a
    for k in range(n - 1):
        ret = multiply(ret, a)

    return ret


@njit
def trimmed(a):

    asize = a.shape[0]

    if asize == 1:
        return a

    trailingzeros = 0
    for i in range(asize):
        if a[asize - i - 1] != 0:
            break
        else:
            trailingzeros += 1
    return a[:asize - trailingzeros]


@njit
def fitsize(a, size):

    asize = a.shape[0]

    if size <= asize:
        return a[:size]

    ret = np.zeros(size)
    ret[:asize] = a
    return ret


@njit
def order(a):
    return trimmed(a).shape[0] - 1


@njit
def align(*args):

    argnum = len(args)

    maxlen = 0
    for polynomial in args:
        plen = polynomial.shape[0]
        if plen > maxlen:
            maxlen = plen

    ret = np.zeros((argnum, maxlen))

    for i in range(argnum):
        for j in range(0, args[i].shape[0]):
            ret[i][j] = args[i][j]

    return ret


@njit
def diff(a):

    asize = a.shape[0]

    if asize <= 1:
        return np.array([0.])

    ret = np.empty(asize - 1)
    for k in range(asize - 1):
        ret[k] = (k + 1) * a[k + 1]
    return ret


@njit
def nthdiff(a, n):

    asize = a.shape[0]
    if asize <= n:
        return np.array([0.])

    for k in range(n):
        a = diff(a)

    return a


@njit
def intg(a):

    asize = a.shape[0]
    ret = np.empty(asize + 1)
    for k in range(asize):
        ret[k + 1] = a[k] / (k + 1)
    return ret


_stirling_const = np.sqrt(2 * np.pi)


@njit
def stirling(n):

    return _stirling_const * np.sqrt(n) * (n / np.e)**n


def tostring(a):

    asize = a.shape[0]

    if asize == 0:
        return "0"

    ret = [format(a[0], "+4")]
    for k in range(1, asize):
        ret.append(format(a[k], "+2"))
        if k == 1:
            ret.append("x")
        else:
            ret.append(f"x^{k}")

    ret = "".join(ret)
    ret = ret.replace("+", " + ")
    ret = ret.replace("-", " - ")

    return ret
