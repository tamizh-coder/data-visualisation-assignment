# data-visualisation-assignment


%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import dataset
iris = datasets.load_iris()
X = iris.data[:2, :]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

plt.figure(2, figsize=(8, 6))
plt.clf()



# plot the first three Principal Component Algorithm dimensions

fig = plt.figure(1, figsize=(15, 15))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.Set1, edgecolor="g", s=280,)
ax.set_title("First three Principal Component Algorithm directions")
ax.set_xlabel("X Axis")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Y Axis")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Z Axis")
ax.w_zaxis.set_ticklabels([])

plt.show()
