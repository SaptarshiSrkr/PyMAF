import sys
import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.neighbors import KernelDensity

from generate import gmm


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

# moving average
y_ma_mean   = []
y_ma_median = []
y_ma_mode   = []
for x in xs:        # x = constant
    xys = [[x,y] for y in ys]
    p_y_gx = np.exp(kde.score_samples(xys))
    y_ma_mean.append(get_mean_pdf(ys, p_y_gx))
    y_ma_median.append(get_median_pdf(ys, p_y_gx))
    y_ma_mode.append(get_mode_pdf(ys, p_y_gx))

dery_ma_mean   = np.diff(y_ma_mean)
dery_ma_median = np.diff(y_ma_median)
dery_ma_mode   = np.diff(y_ma_mode)

# plot
plt.contourf(XX, YY, ZZ, levels=50, cmap="Blues")
plt.plot(samples[:,0], samples[:,1], c="C0", ls="", marker=".", markersize=2, alpha=0.3)
plt.plot(xs, y_ma_mean,   c="red", ls="-",  lw=1)
plt.plot(xs, y_ma_median, c="red", ls="--", lw=1)
plt.plot(xs, y_ma_mode,   c="red", ls=":",  lw=1)
plt.plot(xs[:-1], np.abs(dery_ma_mean)-4,   c="green", ls="-",  lw=1)
plt.plot(xs[:-1], np.abs(dery_ma_median)-4, c="green", ls="--", lw=1)
plt.plot(xs[:-1], np.abs(dery_ma_mode)-4,   c="green", ls=":",  lw=1)
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
plt.savefig(f"figs_samplesize/moving_average_{N_samples}.png", dpi=300, bbox_inches="tight")
