import numpy as np
import wvfunc_gen as wfg
from numpy import sqrt, cos, sin, pi
import graeffe

import matplotlib.pyplot as plt


# TODO:
#   Rewrite but with numpy.polynomial


def crawler(N, L, M, dr, cutoff_fraction):

    radfunc = np.vectorize(wfg.getradial(N, L, M))

    if N == 1:

        _flag = True
        stp_radius = 0
        stp_value = radfunc(0)
        rvalues = [stp_value]
        cutoff_value = stp_value * cutoff_fraction

    else:

        Lcoeffs = wfg._cache_laguerre[N - L - 1, 2 * L + 1]
        rhoLcoeffs = graeffe.pn_shift(Lcoeffs, 1)
        rhoLprimecoeffs = graeffe.pn_shift(wfg._diff(Lcoeffs), 1)
        Dcoeffs = (
            graeffe.pn_add(
                graeffe.pn_add(
                    L * Lcoeffs,
                    -0.5 * rhoLcoeffs
                ),
                rhoLprimecoeffs
            )
        )

        stp_bound = N * graeffe.upperbound(Dcoeffs) / 2 + dr

        crawlpoint = stp_bound
        rvalues = [np.abs(radfunc(crawlpoint))]
        crawlsign = -1

        while crawlsign == -1:

            crawlpoint -= dr
            rvalues.append(np.abs(radfunc(crawlpoint)))
            crawlsign = rvalues[-2] - rvalues[-1]

        rvalues.reverse()

        stp_radius = crawlpoint
        stp_value = rvalues[0]
        cutoff_value = cutoff_fraction * stp_value

        _flag = True

        for i, rv in enumerate(rvalues):
            if rv <= cutoff_value:
                cutoff_radius = stp_radius + dr * (len(rvalues) - 1)
                _flag = False
                break

    if _flag:

        crawlpoint = stp_radius + dr * (len(rvalues) - 1)
        crawlvalue = rvalues[-1]

        while crawlvalue > cutoff_value:

            crawlpoint += dr
            crawlvalue = np.abs(radfunc(crawlpoint))

        cutoff_radius = crawlpoint

    return stp_radius, stp_value, cutoff_radius, cutoff_value


def smartscale(
    N, L, M,
    verbose=False, padding=1, cutoff_fraction=0.5,
):
    """
    Tries to estimate optimal scale and level parameters.
    Uses normalised versions of wave functions, with a0=1.

    Arguments:
        N, L, M: quantum numbers
        verbose=False: whether to print logs
        padding=1.5: tweak parameter for scale
        cutoff_fraction=0.5: tweak parameter for level/lobe size

    Returns: 
        scale, compared to a bounding box of 1.
        level, value you should pass to drawisosurface
        level_radius, the estimated radial distance to the farthest lobe
        stp_radius, the distance to the last extremum
    """

    # HOW IT WORKS:
    #
    #  To return scale, we need to determine how far the orbital extends.
    #  The lobes of the orbital are regions around the extremal points of the
    #  radial part of the wave function. Accordingly, this algorithm finds the
    #  farthest point of the lobe around the farthest stationary point and
    #  uses this to ensure all the lobes are displayed.
    #
    #  The second return value, level, is the value of the probability density
    #  of the wave function at the edge of all the lobes.
    #
    #  ^ |psi|**2
    #  |        ___                                stpoint
    #  |      ./   \.               __                \/     lobe edge
    #  |     /       \            ./  \.              __     /
    #  |    /  lobe   \          /      \           ./  \.  /
    #  |  ./<--------->\. - - -./- - - - \.- - - - / - - -\/ - - - -level
    #  |_/               \.__./            \.____./        \._______
    #  +------------------------------------------------------------> r
    #
    #  It is calculated such that, by default, the farthest lobe of the orbital
    #  has a |psi|**2 half as big on the edge of the lobe as it is at the top
    #  of the lobe. This can be changed by modifying cutoff_fraction.

    harm_norm = wfg._cache_harm_norm[(L, M)]
    Pcoeffs = wfg._cache_legendre[L, M]

    NUMTHETAS = 300

    thetas = np.linspace(0, pi, NUMTHETAS)
    cs = cos(thetas)

    if M % 2 == 1:
        # s = sin(thetas)
        s = 1
    else:
        s = 1

    Pvalues = np.abs(sum(Pcoeffs[k] * cs**k for k in range(len(Pcoeffs))) * s)

    if L == 0:

        aspectratio = 1
        Pbigmax = 1
        Psmallmax = 1

    else:

        Pmaxesindices = np.nonzero(
            np.diff(np.sign(np.diff(Pvalues))))[0]

        Pmaxes = Pvalues[Pmaxesindices]

        top, bottom = Pvalues[0], Pvalues[-1]

        if top > Pvalues[1]:
            # print(f"top {top} {Pvalues[1]}")
            Pmaxes = np.append(Pmaxes, top)
        if bottom > Pvalues[-2]:
            # print(f"top {bottom} {Pvalues[-2]}")
            Pmaxes = np.append(Pmaxes, bottom)

        if Pmaxes.size == 0:

            Pbigmax = Pvalues[0]
            Psmallmax = Pvalues[0]

        else:

            Pbigmax = np.max(Pmaxes)
            Psmallmax = np.min(Pmaxes)

        aspectratio = Pbigmax / Psmallmax

    stp_radius, stp_value, level_radius, cutoff_value = crawler(
        N, L, M, 0.1, cutoff_fraction / aspectratio)

    harmonic_constant = harm_norm * Pbigmax

    level = (
        cutoff_value *
        wfg.normconst(N, L, M) *
        harmonic_constant
    )**2

    scale = 1 / (padding * level_radius)

    if verbose:
        print(f"farthest stp: {stp_radius}\n"
              f"value at stp: {stp_value}\n"
              f"harmonic cst: {harmonic_constant}\n"
              f"scale:        {scale}\n"
              f"level:        {level}")

    # print(f"NLM {N} {L} {M}\n"
    #       f"a.r. {round(aspectratio, 3)} = {Pbigmax}/{Psmallmax}\n"
    #       f"Pmaxes {Pmaxes if L != 0 else None}\n"
    #       f"indices {Pmaxesindices if L != 0 else None}\n")

    # print(f"cutoff: {cutoff_value}\n")

    return scale, level, level_radius, cutoff_value, stp_radius, stp_value
