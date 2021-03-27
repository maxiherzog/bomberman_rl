from sklearn.tree import DecisionTreeRegressor
import numpy as  np
stub = DecisionTreeRegressor(max_depth=1, random_state=0)
X = np.array([[0,0],[1,1]])
y = [4,7]
stub.fit(X,y)
p = stub.predict([[1,0]])

print(p)

import pickle

with open("model/model.pt", "rb") as file:
    regressor = pickle.load(file)

print(len(regressor.forest))
