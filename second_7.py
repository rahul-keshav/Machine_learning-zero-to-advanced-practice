#Linear Model for multiclass classification

from sklearn.datasets import make_blobs
import mglearn
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import numpy as np



X,y= make_blobs(random_state=42)
line=np.linspace(-15,15)
linear_svm=LinearSVC().fit(X,y)
print("coef :\n{}".format(linear_svm.coef_) )
print("intercept:\n",linear_svm.intercept_)
print("coef shape",linear_svm.coef_.shape)
print("intercept shape",linear_svm.intercept_.shape)

#input 47

# mglearn.discrete_scatter(X[:,0],X[:,1],y)
#
# for coef,intercept,color in zip(linear_svm.coef_,linear_svm.intercept_,['b','r','g']):
#     plt.plot(line, ((line*coef[0] +intercept)/-coef[1]),c=color)
#
# plt.ylim(-10,15)
# plt.xlim(-10,8)
# plt.xlabel("Feature:0")
# plt.ylabel("Feature:1")
# plt.legend(["class 0","class 1","class 2"])
# plt.show()

#input 48

mglearn.plots.plot_2d_classification(linear_svm,X,fill=True,alpha=0.7)
mglearn.discrete_scatter(X[:,0],X[:,1],y)


for coef,intercept,color in zip(linear_svm.coef_,linear_svm.intercept_,['b','r','g']):
    Y = (-(line * coef[0] + intercept) / coef[1])
    plt.plot(line, Y,c=color)

plt.xlabel("Feature:0")
plt.ylabel("Feature:1")
plt.legend(["class 0","class 1","class 2"])
plt.show()