

import numpy as np
from numpy import pi, sqrt, cos, sin

import numba

from math import atan2
from cmath import phase

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

from skimage import measure

import wvfunc_gen as wfg
from smartscale import smartscale

import os
from time import time
import winsound
from warnings import warn


# SETUP

# Calculate rainbow ListedColormap

plt.rcParams["mathtext.fontset"] = "dejavuserif"


def sphericals(x, y, z):
    R = x**2 + y**2
    return sqrt(R + z**2), atan2(sqrt(R), z), atan2(y, x)


def rainbowcolor(hue, alpha=1):
    assert 0 <= hue <= 1
    if hue < 1 / 6:
        return np.array([1, 6 * hue, 0, alpha])
    elif 1 / 6 <= hue < 1 / 3:
        return np.array([2 - 6 * hue, 1, 0, alpha])
    elif 1 / 3 <= hue < 1 / 2:
        return np.array([0, 1, -2 + 6 * hue, alpha])
    elif 1 / 2 <= hue < 2 / 3:
        return np.array([0, 4 - 6 * hue, 1, alpha])
    elif 2 / 6 <= hue < 5 / 6:
        return np.array([-4 + 6 * hue, 0, 1, alpha])
    elif 5 / 6 <= hue:
        return np.array([1, 0, 6 - 6 * hue, alpha])


def rainbowcolorvec(hues, alpha=1):
    return np.array([rainbowcolor(hue, alpha) for hue in hues])


colors = rainbowcolorvec(np.linspace(0, 1, 256), 0.2)
rainbow = ListedColormap(colors)


# Functions and setup for drawing surfaces

@numba.vectorize([numba.float64(numba.complex128), numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2


# constants for all the functions


DEFAULT_MIN = -30
DEFAULT_MAX = 30
DEFAULT_NUM = 15
MAXNUM = 101


def getminmaxnum(kwargs):

    if "MIN" in kwargs.keys():
        MIN = kwargs["MIN"]
        del kwargs["MIN"]
    else:
        MIN = DEFAULT_MIN
    if "MAX" in kwargs.keys():
        MAX = kwargs["MAX"]
        del kwargs["MAX"]
    else:
        MAX = DEFAULT_MAX
    if "NUM" in kwargs.keys():
        NUM = kwargs["NUM"]
        del kwargs["NUM"]
    else:
        NUM = DEFAULT_NUM

    return MIN, MAX, NUM


def calculate(func, MIN, MAX, NUM):

    _x, _y, _z = np.meshgrid(
        np.linspace(MIN, MAX, NUM),
        np.linspace(MIN, MAX, NUM),
        np.linspace(MIN, MAX, NUM)
    )

    return np.abs(np.vectorize(func)(_x, _y, _z))**2


def drawisosurf(func, value, ax, verbose=False, allowempty=False, **kwargs):
    """
    draw surface contour

    func: function to draw
    value: what value to draw the contour at
    ax: what axis object to draw to
    """

    MIN, MAX, NUM = getminmaxnum(kwargs)
    DX = (MAX - MIN) / (NUM - 1)

    # if MAXNUM > NUM > 2 * MAXNUM // 3:
    #     warn(f"Large gridnum, {NUM}!")
    # elif NUM >= MAXNUM:
    #     warn(f"Very large gridnum, {NUM}! Did you do this on purpose?")

    _flag_empty = False

    if verbose:
        print("Drawing isosurface.")
        print("\tCalculating volume.")

    vol = calculate(func, MIN, MAX, NUM)

    if verbose:
        print("\tMarching cubes.")

    try:
        verts, faces, _, _ = measure.marching_cubes_lewiner(
            vol, value, spacing=(DX, DX, DX))
    except ValueError:
        if allowempty:
            _flag_empty = True
            if verbose:
                print("Empty")
        else:
            print(f"level. {value}")
            print(f"min. {vol.min()} max. {vol.max()}\n\n")
            raise

    if verbose:
        print(f"\tmin. {vol.min()} max. {vol.max()}\n\n")

    # For some unfathomable reason, the marching cubes function uses
    # the left hand rule, and flips the first two columns of vol.
    # I have no clue why this happens, but it requires me to flip them
    # back.

    if not _flag_empty:

        verts[:, [0, 1]] = verts[:, [1, 0]]

        og = np.repeat(0.5 * (MAX - MIN), 3)
        og = np.repeat([og], len(verts[:, 0]), 0)
        verts -= og
        tsurf = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                                cmap=rainbow,
                                vmin=0, vmax=1)

        if verbose:
            print("\tGenerating facescalars.")

        fc = facescalars(func, faces, verts)

        tsurf.set_array(fc)

    else:

        tsurf = None

    return tsurf


def drawaxis(ax, nx, ny, nz, color="black", linestyle=":", **kwargs):

    MIN, MAX, _ = getminmaxnum(kwargs)
    ax.plot([MIN * nx, MAX * nx],
            [MIN * ny, MAX * ny],
            [MIN * nz, MAX * nz],
            color=color, linestyle=linestyle,
            **kwargs)


def drawbasis(ax, size=10.0, ratio=0.15, color="black", linewidth=0.75, **kwargs):
    ax.quiver3D([0, 0, 0], [0, 0, 0], [0, 0, 0],
                [size, 0, 0], [0, size, 0], [0, 0, size],
                arrow_length_ratio=ratio, color=color, linewidth=linewidth, **kwargs)
    ax.plot([-size, 0], [0, 0], [0, 0], color=color,
            linewidth=linewidth, **kwargs)
    ax.plot([0, 0], [-size, 0], [0, 0], color=color,
            linewidth=linewidth, **kwargs)
    ax.plot([0, 0], [0, 0], [-size, 0], color=color,
            linewidth=linewidth, **kwargs)


def drawunitcircle(ax, SCALE, color="black", linewidth=0.75, **kwargs):
    thetas = np.arange(0, 2 * pi + 0.1, 0.1)
    xs = cos(thetas) * SCALE
    ys = sin(thetas) * SCALE
    zs = np.zeros(len(thetas))
    ax.plot(xs, ys, zs, color=color, linewidth=linewidth, **kwargs)


def centers(triangles):
    return np.mean(triangles, 1)


def facescalars(func, faces, verts):

    cs = centers(verts[faces])

    # coordinates of triangle centers

    xs = cs[:, 0]
    ys = cs[:, 1]
    zs = cs[:, 2]

    def scalar(x, y, z):
        return (phase(func(x, y, z)) / (2 * pi)) % 1

    return (
        np.vectorize(scalar)(xs, ys, zs)
    )


def stamp():
    return round(time())


def prep(axes, fig, titles=["orbitals"], keepbg=False, **kwargs):

    MIN, MAX, _ = getminmaxnum(kwargs)

    MARGIN = 0.1
    fig.subplots_adjust(-MARGIN, -MARGIN,
                        1 + MARGIN, 1 + MARGIN,
                        0.05, 0.05)

    # Create cubic bounding box to simulate equal aspect ratio
    xs = np.array([MIN, MAX])
    ys = np.array([MIN, MAX])
    zs = np.array([MIN, MAX])

    for ax in axes:
        ax.scatter(xs, ys, zs, color=(0, 0, 0, 0))

    # Titles and such

    supfont = {
        "fontname": "Georgia",
        "fontsize": 20
    }

    subfont = {
        "fontname": "Georgia",
        "fontsize": 16
    }

    fig.suptitle(titles.pop(0), weight="bold", **supfont)

    for i in range(len(titles)):
        try:
            axes[i].set_title(titles[i], **subfont)
        except KeyError:
            warn("Too many titles!")

    bright = 0.98

    for ax in axes:
        ax.set_xticks([MIN, (MIN + MAX) / 2, MAX])
        ax.set_yticks([MIN, (MIN + MAX) / 2, MAX])
        ax.set_zticks([MIN, (MIN + MAX) / 2, MAX])

        ax.set_xlim(MIN, MAX)
        ax.set_ylim(MIN, MAX)
        ax.set_zlim(MIN, MAX)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.set_xlabel("x", labelpad=-10)
        ax.set_ylabel("y", labelpad=-10)
        ax.set_zlabel("z", labelpad=-10)

        ax.w_xaxis.set_pane_color((bright, bright, bright, 1))
        ax.w_yaxis.set_pane_color((bright, bright, bright, 1))
        ax.w_zaxis.set_pane_color((bright, bright, bright, 1))

        if not keepbg:
            ax.set_axis_off()

        drawbasis(ax, size=MAX * 1.3, ratio=0.07)


print("Initial setup done.")
print()


#
# ===================================================================
#


# ============
# MAIN_FIGURES
# ============


def drawbohrradii(ax, scale):

    def wholecircle(factor):
        drawunitcircle(ax, scale * 10**factor)

        if i < 3:
            ax.text(0, scale * 10**factor - 0.02, -0.06,
                    f"${int(10**factor)}$", fontsize=8)
        else:
            ax.text(0, scale * 10**factor - 0.01, -0.06,
                    f"$ 10^{{{int(factor)}}} $", fontsize=8)

    def halfcircle(factor):
        drawunitcircle(ax, scale * 10**factor, linestyle="--")

    biggest = sqrt(2) / scale
    upper = np.log10(biggest)
    lower = max(0, round(upper) - 1)

    for i in np.arange(lower, max(1, upper), 1):
        wholecircle(i)
    for i in np.arange(0.5 + lower, upper, 1):
        halfcircle(i)


def main_nlm(N, L, M, *args,
             show=True, save=False, gridnum=21, verbose=False,
             gen=50, block=True, subpath=None, smsctest=False, keepbg=False,
             viewangle=(30, 60), allowempty=False, setlevel=None, setscale=None):

    if verbose:
        print(
            f"========\n"
            f"main_nlm\n"
            f"--------\n"
            f"{args}\n"
            f"{N}, {L}, {M}\n"
            f"show={show}\n"
            f"save={show}\n"
            f"gridnum={gridnum}\n"
            f"verbose={verbose}\n"
            f"========\n\n\n"
        )

    drawcos = False
    drawsin = False

    if M != 0:
        if "cos" in args and "sin" not in args:
            drawcos = True
        elif "cos" not in args and "sin" in args:
            drawsin = True
        elif "cos" in args and "sin" in args:
            raise ValueError("Can't have cos and sin at the same time.")

    # Generate wvfuncs

    if verbose:
        print("Generating functions and calculating scaling.")

    if gen != 0:
        wfg.generate(gen)

    psi = wfg.getpsi(N, L, M, normalised=True)
    psistar = wfg.getpsi(N, L, -M, normalised=True)

    SCALE, LEVEL, level_radius, level_value, stp, _ = smartscale(
        N, L, M, verbose=verbose)

    if setlevel is not None:
        LEVEL = setlevel

    if setscale is not None:
        SCALE = setscale

    if not drawcos and not drawsin:
        def func(x, y, z):
            sp = sphericals(x, y, z)
            sp = (sp[0] / SCALE, sp[1], sp[2])
            return psi(*sp)
    elif drawcos:
        def func(x, y, z):
            sp = sphericals(x, y, z)
            sp = (sp[0] / SCALE, sp[1], sp[2])
            return (psi(*sp) + psistar(*sp)) / sqrt(2)
    elif drawsin:
        def func(x, y, z):
            sp = sphericals(x, y, z)
            sp = (sp[0] / SCALE, sp[1], sp[2])
            return (psi(*sp) - psistar(*sp)) / sqrt(2) / 1j

    # Set up the plot and axes

    if verbose:
        print("Setting up pyplot.")

    fig = plt.figure(figsize=(9, 9))
    mykwargs = {
        "proj_type": "persp",
        "azim": viewangle[0],
        "elev": viewangle[1]
    }

    ax1 = fig.add_subplot(1, 1, 1, projection="3d", **mykwargs)
    axes = [ax1]

    # ----------
    # Parameters
    # ----------

    MYMAX = 1
    MYNUM = gridnum

    # ----------

    if not drawcos and not drawsin:
        titlestring = f"$ \\psi_{{{N}, {L}, {M}}} $"
    elif drawcos:
        titlestring = (f"$ \\frac{{1}}{{\\sqrt{{2}}}}"
                       f" (\\psi_{{{N}, {L}, {{+{abs(M)}}}}} "
                       f"+ \\psi_{{{N}, {L}, {{-{abs(M)}}}}}) $")
    elif drawsin:
        titlestring = (f"$ \\frac{{1}}{{\\sqrt{{2}}}}"
                       f" (\\psi_{{{N}, {L}, {{+{abs(M)}}}}} "
                       f"- \\psi_{{{N}, {L}, {{-{abs(M)}}}}}) $")

    prep(
        axes, fig, MAX=MYMAX, MIN=-MYMAX,
        titles=[
            titlestring,
            ""
        ], keepbg=keepbg
    )

    # ------------------
    # DRAWING STUFF HERE
    # ------------------

    if verbose:
        print("Drawing")
        print("---------------------")
    drawisosurf(
        func, LEVEL, ax1,
        NUM=MYNUM, MIN=-MYMAX, MAX=MYMAX, verbose=verbose,
        allowempty=allowempty
    )

    if verbose:
        print("Drawing other elements.")

    # Bohr radii
    drawbohrradii(ax1, SCALE)

    # Watermark. If you remove this (e.g. )
    ax1.text(1, -.25, -.25, "$\\frac{Mov}{pasd}$",
             fontsize=18, color=(0, 0, 0, 0.08))

    if smsctest:

        ax1.scatter(
            [0],
            [stp * SCALE],
            [0],
            "b"
        )

        ax1.scatter(
            [0],
            [level_radius * SCALE],
            [0],
            "r"
        )

        rs = np.linspace(0, 1 / SCALE, 100)
        scaledrs = np.linspace(0, 1, 100)
        vs = np.vectorize(wfg.getradial(N, L, M))(rs)
        vs = 0.5 * vs / np.max(vs)

        ax1.plot(np.zeros(100), scaledrs, vs, "black")

    if verbose:
        print("---------------------")

    # -----------------

    # File handling and such

    if save:

        if verbose:
            print("File handling.")

        paramstr = ""

        if M == 0:
            Mstr = "0"
        elif M > 0:
            Mstr = f"p{abs(M)}"
        elif M < 0:
            Mstr = f"m{abs(M)}"

        if drawcos:
            paramstr += "-cos"
            Mstr = f"{abs(M)}"
        if drawsin:
            paramstr += "-sin"
            Mstr = f"{abs(M)}"

        # paramstr += f"-gn{gridnum}"

        path = "results/photos"
        if subpath is not None:
            path += f"/{subpath}"

        if not os.path.exists(path):
            os.mkdir(path)

        plt.savefig(
            f"{path}/psi-{N}-{L}-{Mstr}{paramstr}.png",
            format="png"
        ) # -{stamp()}

        plt.close(fig)

    if show:

        if verbose:
            print("Displaying.")

        plt.show(block=block)

    if verbose:
        print("FINISHED!\n\n\n\n")

    #
    # Hacky fix to a problem
    #

    return SCALE, LEVEL


def main():

    def submain(N, L, M, *args, **kwargs):

        return main_nlm(
            N, L, M, *args,
            viewangle=(15, 30),
            gen=0,
            gridnum=101,
            verbose=True,
            show=False,
            save=True,
            subpath="tests/profiling",
            keepbg=False,
            allowempty=True,
            **kwargs
        )

    wfg.generate(80)

    level = 0

    submain(35, 12, 4)

    raise Exception

    for N in range(1, 80):

        for L in range(N):

            for M in range(0, L + 1):

                print(N, L, M)

                if M == 0:

                    scale, level = submain(N, L, M)

                else:

                    submain(N, L, +M)
                    submain(N, L, -M)

                    submain(N, L, M, "cos", setscale=scale, setlevel=level)
                    submain(N, L, M, "sin", setscale=scale, setlevel=level)


if __name__ == "__main__":

    import cProfile
    cProfile.run("main()"`, "profiling/profileresults")


# # ============
# # MAIN ANIMATE
# # ============


# def main_animate():

#     fig = plt.figure(figsize=(9, 9))
#     mykwargs = {
#         "proj_type": "persp",
#         "azim": 45,
#         "elev": 30
#     }

#     ax1 = fig.add_subplot(1, 1, 1, projection="3d", **mykwargs)

#     axes = [ax1]

#     MARGIN = 0
#     SPACING = 0
#     fig.subplots_adjust(left=MARGIN, right=1 - MARGIN,
#                         bottom=MARGIN, top=1 - MARGIN,
#                         wspace=SPACING, hspace=SPACING)

#     def timedep(n):

#         phase = n * pi / 16
#         dx = 0
#         dz = 0
#         if n < 120:
#             dy = 30 - 30 * n / 120
#         if n >= 120:
#             dy = 0

#         return phase, dx, dy, dz

#     prep(axes, fig)
#     ax1.text(
#         0, 0, 25,
#         # "$ \mathrm{Re}(\psi_{3,2,+1}) $"
#         # " $ \mathrm{Re}(3d_{xz}) $ "
#         ""
#     )

#     def nthfunc(n):

#         phase, dx, dy, dz = timedep(n)

#         def func(x, y, z):
#             sp1 = sphericals(x - dx, y - dy, z - dz)
#             sp2 = sphericals(x + dx, y + dy, z + dz)

#             return (
#                 (orb2px(*sp1) * 1j * exp(1j * phase)) +
#                 (orb2py(*sp2) * exp(1j * phase))
#             ).real
#         return func

#     surfs = []

#     MYNUM = 25

#     def update(num):

#         print("=", end="")

#         surf = surfs.pop()
#         if surf is not None:
#             surf.remove()
#         surfs.append(drawisosurf(nthfunc(num), 5e-5, ax1,
#                                  allowempty=True, NUM=MYNUM))

#     def init():

#         surfs.append(drawisosurf(nthfunc(0), 5e-5, ax1,
#                                  allowempty=True, NUM=MYNUM))

#     FRAMENUMBER = 200

#     anim = FuncAnimation(fig, update,
#                          init_func=init,
#                          frames=np.arange(0, FRAMENUMBER), interval=50)
#     anim.save(
#         f"results/vids/{stamp()}.mp4", dpi=80, writer="ffmpeg")

#     winsound.Beep(600, 500)
