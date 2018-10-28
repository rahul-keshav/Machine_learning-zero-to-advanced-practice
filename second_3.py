from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import mglearn

###################################
#Input 24
##################################
X,y=mglearn.datasets.make_wave(n_samples=40)
#split the wave dataset into a training anda test set
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

fig,axes=plt.subplots(1,3, figsize=(15,4))

#create 1000 data points, evenly spaced between -3 to +3
line = np.linspace(-3,3,1000).reshape(-1,1)
# X_train,X_test,y_train,y_test=train_test_split()

for n_neighbors, ax in zip([1,3,9],axes):
    #make prediction using 1,3 or 9 neighbors
    reg=KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train,y_train)
    ax.plot(line,reg.predict(line))
    ax.plot(X_train,y_train,'^',c=mglearn.cm2(0),markersize=8)
    ax.plot(X_test,y_test,'v',c=mglearn.cm2(1),markersize=8)
    ax.set_title(
        "{} neighbors(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors,reg.score(X_train,y_train),reg.score(X_test,y_test)
        )
    )
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model Predictions","Training data/target","Test data/target"],loc="best")
plt.show()