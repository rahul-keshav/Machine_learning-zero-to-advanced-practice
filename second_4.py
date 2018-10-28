import mglearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#input 25

mglearn.plots.plot_linear_regression_wave()
plt.show()


#input 26
X,y=mglearn.datasets.make_wave(n_samples=60)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

reg=LinearRegression()
reg.fit(X_train,y_train)
#input 27
print("cofficient=%s,intercept=%s"%(reg.coef_,reg.intercept_))
#input 28
print("taining set score={}".format(reg.score(X_train,y_train)))
print("test set score={}".format(reg.score(X_test,y_test)))
























