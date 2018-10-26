from sklearn.datasets import load_breast_cancer,load_boston
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# X,y = mglearn.datasets.make_forge()
# mglearn.discrete_scatter(X[:,0],X[:,1],y)
# plt.legend(["Class 0", "Class 1"], loc=4)
# plt.xlabel("First feature")
# plt.ylabel("Second feature")
# print("X.shape :{}".format(X.shape))
# plt.show()


# X,y = mglearn.datasets.make_wave(n_samples=40)
# plt.plot(X,y,'o')
# plt.ylim(-3,3)
# plt.xlabel("feature")
# plt.ylabel("target")
# plt.show()
#
# cancer=load_breast_cancer()
# print("Sample counts per class:\n{}".format({n: v for n,v in zip(cancer.target_names,np.bincount(cancer.target)) } ))
#

# boston =load_boston()
# print("feature name : {}".format(boston['feature_names']))
# print("keys name : {}".format(boston.keys()))
# print("target name : {}".format(boston['target']))
# print("target name : {}".format(boston['DESCR']))

# X,y =mglearn.datasets.load_extended_boston()
# print("X.shape:{}".format(X.shape))


# mglearn.plots.plot_knn_classification(n_neighbors=25)
# plt.show()

########################################################
######################################################
#
# X,y = mglearn.datasets.make_forge()
# X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
#
# clf=KNeighborsClassifier(n_neighbors=3)
# clf.fit(X_train,y_train)
# print("test pridiction={}".format(clf.predict(X_test)))
# print("xtrainn={}",X_train)
# print("ytrain={}",y_train)
# print("x test",X_test)
# print("y test",y_test)
# print("score",clf.score(X_test,y_test))

#############################################################
# input 17
#############################################################

# fig, axes=plt.subplots(1,3, figsize=(10,3))
#
# for n_neighbors,ax in zip([1,3,9],axes):
#     clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
#     mglearn.plots.plot_2d_separator(clf,X,fill=True,eps=0.5,ax=ax,alpha=0.4)
#     mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
#     ax.set_title=("{} neighbor(s)".format(n_neighbors))
#
#     ax.set_xlabel("feature 0")
#     ax.set_ylabel("feature 1")
# axes[0].legend(loc=3)
# plt.show()

##############################################################
#  input 18
##############################################################
cancer = load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=66)

training_accuracy=[]
test_accuracy=[]

neighbors_settings=range(1,11)

for n_neighbors in neighbors_settings:
    clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings,training_accuracy,label="training accuracy")
plt.plot(neighbors_settings,test_accuracy, label="test accuracy")
plt.ylabel("accuracy")
plt.xlabel("neighbors")
# plt.legend()
plt.show()
