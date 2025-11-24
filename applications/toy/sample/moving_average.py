import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from generate import gmm


def get_mode_sample(x):
    """get mode for sample xs"""
    hist, bin_edges = np.histogram(x, bins='auto')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    max_bin_index = np.argmax(hist)
    return bin_centers[max_bin_index]
    # values, counts = np.unique(x, return_counts=True)
    # idx = np.argmax(counts)
    # return values[idx]

def get_mean_sample(x):
    """get mean for a sample xs"""
    if x.shape[0] == 0:
        return np.nan
    return np.mean(x)

def get_median_sample(x):
    """get median for a sample xs"""
    if x.shape[0] == 0:
        return np.nan
    return np.median(x)

def get_sample_given_x(samples, x, delta_x):
    """get samples for given x within [x-delta_x, x+delta_x]"""
    x_lower = x - delta_x
    x_upper = x + delta_x
    samples_x = samples[(samples[:,0]>=x_lower) & (samples[:,0]<x_upper), :]
    return samples_x

# params
N = 100     # decides the binsize
N_sample = 1_00_000
xrange = [-5, 15]
yrange = [-5, 15]

# samples
samples, _ = gmm.sample(N_sample)

# space discretization
xs = np.linspace(*xrange, N)
ys = np.linspace(*yrange, N)
delta_x = (xrange[1]-xrange[0])/(2*N)

# moving average
y_ma_mean   = []
y_ma_median = []
y_ma_mode   = []
for x in xs:        # x = constant
    # get samples for given x
    samples_x = get_sample_given_x(samples, x, delta_x)
    # get statistics
    y_ma_mean.append(get_mean_sample(samples_x[:,1]))
    y_ma_median.append(get_median_sample(samples_x[:,1]))
    y_ma_mode.append(get_mode_sample(samples_x[:,1]))

dery_ma_mean   = np.diff(y_ma_mean)
dery_ma_median = np.diff(y_ma_median)
dery_ma_mode   = np.diff(y_ma_mode)

# # plot
plt.plot(samples[:,0], samples[:,1], ls="", marker=".", ms=1, c="lightgray", alpha=0.3)
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
plt.savefig("figs/moving_average.png", dpi=300, bbox_inches="tight")
