import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.neighbors import KernelDensity
from denmarf import DensityEstimate

from generate import gmm


#| control parameters
gmm_dist_type = 'well_separated_peaks'   # 'well_separated_peaks' or 'not_well_separated_peaks'
de_method = 'nfde'   # 'kde' or 'nfde'
seed = 6333     # nfde

np.random.seed(seed)
DE_outdir = Path(gmm_dist_type, de_method, "DEs")
fig_outdir = Path(gmm_dist_type, de_method, "figs")
DE_outdir.mkdir(parents=True, exist_ok=True)
fig_outdir.mkdir(parents=True, exist_ok=True)


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
if gmm_dist_type == 'well_separated_peaks':
    xrange = [-5, 15]
    yrange = [-5, 15]
elif gmm_dist_type == 'not_well_separated_peaks':
    xrange = [-5, 10]
    yrange = [-5, 10]

# samples
samples, _ = gmm.sample(N_samples)

# PDF from samples
de_file = Path(DE_outdir, f"de_{N_samples}.pkl")
print("Checking for existing density estimate...")
if de_file.exists():
    de = torch.load(de_file, weights_only=False)
    print("Found and loaded density estimate.")
else:
    print(f"not found, computing density estimate for {N_samples} samples...")
    if de_method == 'kde':
        de = KernelDensity(kernel='gaussian', bandwidth='scott').fit(samples)
    elif de_method == 'nfde':
        de = DensityEstimate(seed=seed).fit(samples)
        de.save(de_file)
        print("Genrated and saved to ", de_file)
    else:
        raise ValueError(f"Unknown density estimate method: {de_method}")

# grid
xs = np.linspace(*xrange, N)
ys = np.linspace(*yrange, N)
XX, YY = np.meshgrid(xs, ys)
grid = np.column_stack([XX.ravel(), YY.ravel()])

# PDF
z_grid = de.score_samples(grid)
ZZ = z_grid.reshape(XX.shape)

# KL divergence along x
KL_div_vals = []
# 1st value
xys = np.array([[xs[0],y] for y in ys])
p_y_gx = [np.exp(de.score_samples(xys))]
for x in xs[1:]:        # x = constant
    xys = np.array([[x,y] for y in ys])
    p_y_gx.append(np.exp(de.score_samples(xys)))
    KL_div_vals.append(
        KL_div(p_y_gx[-2], p_y_gx[-1])
        )

# plot
xs_centered = 0.5 * (xs[1:] + xs[:-1])
plt.contourf(XX, YY, ZZ, levels=50, cmap="Blues")
plt.plot(samples[:,0], samples[:,1], c="C0", ls="", marker=".", markersize=2, alpha=0.3)
plt.plot(xs_centered, KL_div_vals/np.max(KL_div_vals) * 5,   c="red", ls="-",  lw=1, label="KL divergence (scaled)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc=2)
plt.title("A toy dist - gaussian mixture model")
plt.savefig(Path(fig_outdir, f"KLDiv_{N_samples}.png"), dpi=300, bbox_inches="tight")
print("Saved figure to ", Path(fig_outdir, f"KLDiv_{N_samples}.png"))
