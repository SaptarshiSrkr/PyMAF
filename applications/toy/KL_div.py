from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

# i = 58
# p_y_gx1 = np.exp(gmm.score_samples([[xs[i+0],y] for y in ys]))
# p_y_gx2 = np.exp(gmm.score_samples([[xs[i+1],y] for y in ys]))
# KL_dv_val = KL_div(p_y_gx1, p_y_gx2)

# plt.plot(xs, p_y_gx1, label="x="+str(xs[i+0]))
# plt.plot(xs, p_y_gx2, label="x="+str(xs[i+1]))
# plt.xlabel("y")
# plt.ylabel("p(y|x)")
# plt.title("KL_divergence = {:.4f}".format(KL_dv_val))
# plt.legend()
# plt.show()

KL_div_vals = []
# 1st value
xys = [[xs[0],y] for y in ys]
p_y_gx = [np.exp(gmm.score_samples(xys))]

for x in xs[1:]:        # x = constant
    xys = [[x,y] for y in ys]
    p_y_gx.append(np.exp(gmm.score_samples(xys)))
    KL_div_vals.append(
        KL_div(p_y_gx[-2], p_y_gx[-1])
        )

# plot
plt.contourf(XX, YY, ZZ, levels=50, cmap="Blues")
plt.plot(xs[1:], KL_div_vals/np.max(KL_div_vals) * 5,   c="red", ls="-",  lw=1, label="KL divergence (scaled)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc=2)
plt.title("A toy dist - gaussian mixture model")
plt.savefig("figs/KLDiv2.svg", bbox_inches="tight")
