import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# TODO: remove magic numbers

X = [
       [-2, 4], [-2, 2], [-4, 1],
       [ 1,-1], [ 2,-2], [ 1,-3]
    ]
y = [ 1, 1, 1, 0, 0, 0]

# numpy style list is used in library
X = np.array(X)
y = np.array(y)
z = np.zeros(6)

clf = SVC(kernel="linear", C=1000)
clf.fit(X, y)

v = clf.coef_[0]
b = clf.intercept_[0]

print(v)
print(b)
    
for i in range(6):
    x = X[i]
    p = v[0] * x[0] + v[1] * x[1]
    z[i] = p + b
    print("point ", i+1, " : ", z[i])

# try to find minimum value in each classes
tmp = z[:3]
tmp = abs(tmp)
index_1 = tmp.argmin()

tmp = z[3:6]
tmp = abs(tmp)
index_2 = 3 + tmp.argmin()

print(index_1)
print(index_2)

# find intercept in each lines
x = X[index_1] # select support vector
p = v[0] * x[0] + v[1] * x[1]
q = - (p + b)

# y = -a/b x - c/b
k = -v[0]/v[1]
c = -q   /v[1]
print("k : ", k, " c : ", c)

x = X[index_2] # select support vector
p = v[0] * x[0] + v[1] * x[1]
q = - (p + b)

# y = -a/b x - c/b
k = -v[0]/v[1]
c = -q   /v[1]
print("k : ", k, " c : ", c)


## visualization
#plt.scatter(X[:, 0], X[:, 1], c=y, s=4, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(
    XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
)
# plot support vectors
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()




