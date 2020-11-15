
import functions.legendre as legendre
import functions.legendrenojit as legendrenojit
import numpy as np
from time import time

legendre.asc_coeffs(0)

t = time()
for l in range(250):
    print(l, end=" ")
    a = legendre.asc_coeffs(l)
print("\njit", time() - t)