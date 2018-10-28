import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split



# mglearn.plots.plot_knn_regression(n_neighbors=1)
# mglearn.plots.plot_knn_regression(n_neighbors=3)
# plt.show()

###################################
#Input 21
##################################
X,y=mglearn.datasets.make_wave(n_samples=1000)
#split the wave dataset into a training anda test set
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

reg= KNeighborsRegressor(n_neighbors=100)
reg.fit(X_train,y_train)
print("test pridiction :\n{}".format(reg.predict(X_test)))
print("accuracy",reg.score(X_test,y_test))
plt.scatter(X_train,y_train)
plt.scatter(X_test,y_test)
plt.show()