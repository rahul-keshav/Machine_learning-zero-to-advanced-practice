from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


iris_dataset=load_iris()
# print(iris_dataset.keys())
# print(iris_dataset.DESCR)
# print('Keys of iris_dataset:\n{}'.format(iris_dataset.keys()))
# print(iris_dataset['DESCR'][:193]+'\n...')
# print("target name : {}".format(iris_dataset['target_names']))
# print("features are :{}".format(iris_dataset['feature_names']))
# print("type of data is :{}".format(type(iris_dataset['data'])))
# print("shape of data is :{}".format(iris_dataset['data'].shape))
# print("first five data are:\n{}".format(iris_dataset['data'][:5]))
# print("Type of target :{}".format(type(iris_dataset['target'])))
# print("shope of target :{}".format(iris_dataset['target'].shape))
# print("target:\n{}".format(iris_dataset['target']))

X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
# print("X_train:\n{}".format(X_train))
# print("X_train shape :\n{}".format(X_train.shape))
# print("y_train shape :\n{}".format(y_train.shape))
# print("X_test shape :\n{}".format(X_test.shape))
# print("y_test shape :\n{}".format(y_test.shape))


# iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset.feature_names)
# grr=pd.scatter_matrix(iris_dataframe,
#                       c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},
#                       s=60,alpha=.8,cmap=mglearn.cm3)
# plt.show()

knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
# X_new=np.array([[5,2.9,1,0.2]])
# print("X_new shape:\n{}".format(X_new.shape))
# prediction=knn.predict(X_new)
# print("prediction:{}".format(prediction))
# print("prediction target name:{}".format(iris_dataset['target_names'][prediction]))

y_pred=knn.predict(X_test)
print("y_pred:\n{}".format(y_pred))
print(knn.score(X_test,y_test))












