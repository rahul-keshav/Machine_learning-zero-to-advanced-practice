import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd

x=np.array([[1,2,3], [4,5,6]])
print("x:\n{}".format(x))

# creat a 2d numpy array with a diagonal of ones and zeros everywhere else
eye = np.eye(4)
print("Numpy array \n{}".format(eye))

# conver the numpy array to a scipy sparse matrix in csr format
# only the nonzero entries are stored
sparse_matrix = sparse.csr_matrix(eye)

print("\n Scipy sparse csr matrix:\n{}".format(sparse_matrix))

# %matplotlib inline
# matplotlib.pylot example

# x=np.linspace(-10,10,100)
# y=np.sin(x)
# axis=np.linspace(0,0,100)
# plt.plot(x,y,marker='x')
# plt.plot(x,axis)
# plt.show()


# pandas example
data={'name':['john','Anna','peter','Linda'],
      'location':['newyork','paris','berlin','london'],
      'age':[24,13,53,33]
      }

data_pandas=pd.DataFrame(data)


from IPython.display import display
display(data_pandas)
display(data_pandas[data_pandas.age >30])