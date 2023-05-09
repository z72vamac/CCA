import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-poster')

fig = plt.figure()
ax = plt.axes(projection='3d')
#
# x = np.arange(0, 34)
# y = np.arange(0, 34)
#
# X, Y = np.meshgrid(x, y)

# Z = np.genfromtxt('pareto_test.csv', delimiter=',')
#
# Z = Z[0:25, 0:25]
#
# sns.heatmap(Z[::-1, :], annot=True)
#
# Z = np.genfromtxt('pareto_train.csv', delimiter=',')

x = np.arange(0, 6)
y = np.arange(0, 5)

X, Y  = np.meshgrid(x, y)

Z = np.genfromtxt('paretowine_train.csv', delimiter=',')

Z = Z[0:6, 0:5]

# sns.heatmap(Z[::-1, :], xticklabels=[1, 2, 3, 4, 5], yticklabels=[6, 5, 4, 3, 2, 1], annot=True)

# plt.show()


# Z = np.genfromtxt('w1s.csv', delimiter=",") != 0

# sns.heatmap(Z)
# plt.show()

# Z = np.nan_to_num(Z, nan=np.nanmean(Z))
#
surf = ax.plot_surface(X, Y, Z[X, Y], cmap = plt.cm.cividis)

ax.set_xticks([1, 2, 3, 4, 5, 6])
ax.set_xlabel('x', labelpad=20)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()
