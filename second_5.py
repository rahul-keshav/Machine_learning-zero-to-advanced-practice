import mglearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import train_test_split

#input 29

X,y=mglearn.datasets.load_extended_boston()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
reg=LinearRegression()
reg.fit(X_train,y_train)

#input 30

print("Training set score:{:.2f}".format(reg.score(X_train,y_train)))
print("Test set score:{:.2f}".format(reg.score(X_test,y_test)))

# input 31

ridge= Ridge().fit(X_train, y_train)

print("Ridge Training set score:{:.2f}".format(ridge.score(X_train,y_train)))
print("Ridge Test set score:{:.2f}".format(ridge.score(X_test,y_test)))

# input 32

ridge10=Ridge(alpha=10).fit(X_train,y_train)

print("Ridge10 Training set score:{:.2f}".format(ridge10.score(X_train,y_train)))
print("Ridge10 Test set score:{:.2f}".format(ridge10.score(X_test,y_test)))

# input 33

ridge01=Ridge(alpha=0.1).fit(X_train,y_train)

print("Ridge01 Training set score:{:.2f}".format(ridge01.score(X_train,y_train)))
print("Ridge01 Test set score:{:.2f}".format(ridge01.score(X_test,y_test)))

# input 34

plt.plot(ridge.coef_,'s',label="Ridge alpha = 1")
plt.plot(ridge10.coef_,'^',label='Ridge alpha = 10')
plt.plot(ridge01.coef_,'v',label='Ridge alpha = 0.1')

plt.plot(reg.coef_,'o',label='linearRegression')
plt.xlabel("coefficient index")
plt.ylabel("cofficient magnitude")
plt.hlines(0,0,len(reg.coef_))
plt.ylim(-25,25)
plt.legend()
plt.show()
# input 35
mglearn.plots.plot_ridge_n_samples()
plt.show()

