import sys
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def get_mode_pdf(x, p):
    """get mode for a PDF p(x)"""
    idx = np.argmax(p)
    return x[idx]

def get_mean_pdf(x, p):
    """get mean for a PDF p(x)"""
    mean = trapezoid(x*p, x) / trapezoid(p, x)
    return mean

def get_median_pdf(x, p):
    """get median for a PDF p(x)"""
    cdf = np.cumsum(p)
    cdf = cdf / cdf[-1]
    idx = np.searchsorted(cdf, 0.5)
    return x[idx]


# params
N = 100     # x space discretization
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

# moving average
y_ma_mean   = []
y_ma_median = []
y_ma_mode   = []
for x in xs:        # x = constant
    xys = np.array([[x,y] for y in ys])
    p_y_gx = np.exp(de.score_samples(xys))
    y_ma_mean.append(get_mean_pdf(ys, p_y_gx))
    y_ma_median.append(get_median_pdf(ys, p_y_gx))
    y_ma_mode.append(get_mode_pdf(ys, p_y_gx))

dery_ma_mean   = np.diff(y_ma_mean)
dery_ma_median = np.diff(y_ma_median)
dery_ma_mode   = np.diff(y_ma_mode)

# plot
xs_centered = 0.5 * (xs[1:] + xs[:-1])
plt.contourf(XX, YY, ZZ, levels=50, cmap="Blues")
plt.plot(samples[:,0], samples[:,1], c="C0", ls="", marker=".", markersize=2, alpha=0.3)
plt.plot(xs, y_ma_mean,   c="red", ls="-",  lw=1)
plt.plot(xs, y_ma_median, c="red", ls="--", lw=1)
plt.plot(xs, y_ma_mode,   c="red", ls=":",  lw=1)
plt.plot(xs_centered, np.abs(dery_ma_mean)-4,   c="green", ls="-",  lw=1)
plt.plot(xs_centered, np.abs(dery_ma_median)-4, c="green", ls="--", lw=1)
plt.plot(xs_centered, np.abs(dery_ma_mode)-4,   c="green", ls=":",  lw=1)
plt.xlabel("x")
plt.ylabel("y")
legends = [
    Line2D([],[], color="black", ls="-",  lw=1, label="mean"),
    Line2D([],[], color="black", ls="--", lw=1, label="median"),
    Line2D([],[], color="black", ls=":",  lw=1, label="mode"),
    Line2D([],[], color="red",   ls="-",  lw=1, label="MA"),
    Line2D([],[], color="green", ls="-",  lw=1, label="derivative of MA"),
]
plt.legend(handles=legends, loc=2)
plt.title("A toy dist - gaussian mixture model")
plt.savefig(Path(fig_outdir, f"moving_average_{N_samples}.png"), dpi=300, bbox_inches="tight")
print("Saved figure to ", Path(fig_outdir, f"moving_average_{N_samples}.png"))
