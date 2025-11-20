import numpy as np
import matplotlib.pyplot as plt

from generate import gmm


# params
N = 100
xrange = [-5, 15]
yrange = [-5, 15]

# grid
xs = np.linspace(*xrange, N)
ys = np.linspace(*yrange, N)
XX, YY = np.meshgrid(xs, ys)
grid = np.column_stack([XX.ravel(), YY.ravel()])

# PDF
z_grid = gmm.score_samples(grid)
ZZ = z_grid.reshape(XX.shape)

# moving average
y_ma = []
for x in xs:        # x = constant
    xys = [[x,y] for y in ys]
    p_y_gx = gmm.score_samples(xys)
    p_y_gx_max_idx = np.argmax(p_y_gx)
    y_ma.append(ys[p_y_gx_max_idx])

# plt.scatter(grid[:,0], grid[:,1])
# plt.show()

# # plot
plt.contourf(XX, YY, ZZ, levels=50)
plt.plot(xs, y_ma)
plt.xlabel("x")
plt.ylabel("y")
plt.title("A toy dist - gaussian mixture model")
plt.savefig("moving_average.png", bbox_inches="tight")
