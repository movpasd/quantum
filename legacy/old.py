# OLD CODE


# Draw quivers
R = 15.0
N = 20
positions = np.array([np.array([R * cos(th), R * sin(th), 0])
                      for th in np.linspace(0, 2 * pi, N)])
ups = np.repeat([[0, 0, 1]], len(positions), 0)
vecs = np.cross(ups, positions / R)
ax1.quiver(
    positions[:, 0], positions[:, 1], positions[:, 2],
    vecs[:, 0], vecs[:, 1], vecs[:, 2],
    length=2.5, pivot="middle", color=(0, 0, 0)
)