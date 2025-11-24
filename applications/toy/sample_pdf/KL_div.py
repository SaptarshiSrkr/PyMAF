import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from generate import gmm


def KL_div(p, q):
    """Kullback-Leibler divergence D_KL(P||Q) for discrete distributions.

    Args:
        p: list or np.array, P distribution
        q: list or np.array, Q distribution
    Returns:
        D_KL(P||Q)
    """
    # Normalize
    p = np.asarray(p) / np.sum(p)
    q = np.asarray(q) / np.sum(q)

    # Add small epsilon to avoid log(0)
    eps = 1e-16     # machine precision for float64
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)

    return np.sum(p * np.log(p / q))


# params
N = 100    # x space discretization
N_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 1_00_000
xrange = [-5, 10]
yrange = [-5, 10]

# samples
samples, _ = gmm.sample(N_samples)

# PDF from samples
kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(samples)

# grid
xs = np.linspace(*xrange, N)
ys = np.linspace(*yrange, N)
XX, YY = np.meshgrid(xs, ys)
grid = np.column_stack([XX.ravel(), YY.ravel()])

# PDF
z_grid = kde.score_samples(grid)
ZZ = z_grid.reshape(XX.shape)

# KL divergence along x
KL_div_vals = []
# 1st value
xys = [[xs[0],y] for y in ys]
p_y_gx = [np.exp(kde.score_samples(xys))]
for x in xs[1:]:        # x = constant
    xys = [[x,y] for y in ys]
    p_y_gx.append(np.exp(kde.score_samples(xys)))
    KL_div_vals.append(
        KL_div(p_y_gx[-2], p_y_gx[-1])
        )

# plot
plt.contourf(XX, YY, ZZ, levels=50, cmap="Blues")
plt.plot(samples[:,0], samples[:,1], c="C0", ls="", marker=".", markersize=2, alpha=0.3)
plt.plot(xs[1:], KL_div_vals/np.max(KL_div_vals) * 5,   c="red", ls="-",  lw=1, label="KL divergence (scaled)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc=2)
plt.title("A toy dist - gaussian mixture model")
plt.savefig(f"figs_samplesize/KLDiv_{N_samples}.png", dpi=300, bbox_inches="tight")
