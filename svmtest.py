import numpy as np
import math
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# iris = datasets.load_iris()
# X = iris["data"][:, (2, 3)] # petal length, petal width
# Y = (iris["target"] \== 2).astype(np.float64) # Iris-Virginica
# print(X)
# print(Y)


# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
# SVC - Support vector classification
# LinearSVC ( C: Regularization parameter.
#             loss: specifies loss function

X = [
       [-2, 4], [-2, 2], [-4, 1],
       [ 1,-1], [ 2,-2], [ 1,-3]
    ]

Y = [ 1, 1, 1, -1, -1, -1]

#linear = LinearSVC(C=1000, loss="hinge")
#clf = Pipeline((
#                     ("scaler", StandardScaler()),
#                     ("linear_svc", linear),
#                  ))
clf = SVC(kernel="linear", C=1000)
clf.fit(X, Y)

#v = linear.coef_[0]
#b = linear.intercept_[0]
v = clf.coef_[0]
b = clf.intercept_[0]

print(v)
print(b)

for i in range(6):
    x = X[i]
    p = v[0] * x[0] + v[1] * x[1]
    q = p + b
    print("point ", i+1, " : ", q)

sample = [5.5, 1.7]
norm = math.sqrt(1.63339421 * 1.63339421 + 2.38803113 * 2.38803113)

# sklearn.svm.NuSVC
# Nu-Support Vector Classification.
# Similar to SVC but uses a parameter to control the number of support vectors.
#svm_non = Pipeline((
#                     ("scaler", StandardScaler()),
#                     ("nonlinear_svc", LinearSVC(C=1, loss="hinge")),
#                  ))
#svm_non.fit(X_linear, Y_linear)
