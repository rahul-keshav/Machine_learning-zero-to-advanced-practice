#Linear Model for multiclass classification

from sklearn.datasets import make_blobs
import mglearn
import matplotlib.pyplot as plt
X,y= make_blobs(random_state=42)

mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("Feature:0")
plt.ylabel("Feature:1")
plt.legend(["class 0","class 1","class 2"])
plt.show()
