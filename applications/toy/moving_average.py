import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

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
N = 100
xrange = [-5, 10]
yrange = [-5, 10]

# grid
xs = np.linspace(*xrange, N)
ys = np.linspace(*yrange, N)
XX, YY = np.meshgrid(xs, ys)
grid = np.column_stack([XX.ravel(), YY.ravel()])

# PDF
z_grid = gmm.score_samples(grid)
ZZ = z_grid.reshape(XX.shape)

# moving average
y_ma_mean   = []
y_ma_median = []
y_ma_mode   = []
for x in xs:        # x = constant
    xys = [[x,y] for y in ys]
    p_y_gx = np.exp(gmm.score_samples(xys))
    y_ma_mean.append(get_mean_pdf(ys, p_y_gx))
    y_ma_median.append(get_median_pdf(ys, p_y_gx))
    y_ma_mode.append(get_mode_pdf(ys, p_y_gx))


# # plot
plt.contourf(XX, YY, ZZ, levels=50, cmap="Blues")
plt.plot(xs, y_ma_mean,   c="red", ls="-",  lw=1, label="mean")
plt.plot(xs, y_ma_median, c="red", ls="--", lw=1, label="median")
plt.plot(xs, y_ma_mode,   c="red", ls=":",  lw=1, label="mode")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc=4)
plt.title("A toy dist - gaussian mixture model")
plt.savefig("figs/moving_average3.svg", bbox_inches="tight")
