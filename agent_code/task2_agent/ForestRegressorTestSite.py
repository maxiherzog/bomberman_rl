from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import numpy as np
X, y = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)
regr = RandomForestRegressor(max_depth=2, random_state=0)
Y = np.tile(y,(2,1)).T
Y[:,1] *= 2

regr.fit(X, Y)
#print(X[0],Y[0])
#print(regr.predict([[0, 0, 0, 0]]).T)



regr.fit(X, 2*y)

A = np.reshape(np.arange(12), (6,-1))
print(A.shape)
print( A)
b = np.arange(6)*-1
b = np.reshape(b, (-1, 1))
print(b)
C = np.append(A,b, axis=1)
print(A,b,C)

