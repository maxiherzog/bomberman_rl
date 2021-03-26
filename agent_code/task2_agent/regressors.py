# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:09:57 2021

@author: Philipp
"""
# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.backend import clear_session

from builtins import enumerate


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import numpy as np


class Regressor:
    def __init__(self, n_features, n_actions=6):
        self.n_features = n_features
        self.n_actions = n_actions

    def fit(self, features, values):
        """Use new data to update model. 'features' are vectors with gamestate features AND action as last element."""
        raise NotImplementedError("subclasses must override fit()!")

    def predict(self, features):
        """Predict value vector with an entry for every possible action."""
        raise NotImplementedError("subclasses must override predict()!")

    # def update(self, transitions):
    #    raise NotImplementedError("subclasses must override update()!")
    # this is unused right?


class Network(Regressor):
    def __init__(self, n_features, n_actions=6, n_layers=3, n_epochs=2):

        super().__init__(n_features, n_actions)
        tf.get_logger().setLevel("INFO")
        clear_session()

        self.n_layers = n_layers
        self.n_epochs = n_epochs

        self.model = Sequential()

        # https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
        # create model
        # relu passt?
        # print(n_features)

        self.model.add(
            Dense(
                n_actions,
                activation="relu",
                kernel_initializer="he_normal",
                input_shape=(n_features,),
            )
        )
        for i in range(n_layers - 2):
            self.model.add(
                Dense(n_actions, activation="relu", kernel_initializer="he_normal")
            )
        self.model.add(
            Dense(n_actions, activation="sigmoid", kernel_initializer="he_normal")
        )

        # configures model for training
        self.model.compile(optimizer="adam", loss="mse")

    def fit(self, features, values):

        for k in range(len(features)):
            f = features[k]

            action_index = f[self.n_features]
            f = np.delete(f, self.n_features)

            target = np.zeros(self.n_actions)

            for i in range(self.n_actions):
                if i != action_index:
                    target[i] = self.predict(f)[0][i]

            target[action_index] = values[k]
            with tf.device("/device:GPU:0"):

                self.model.fit(
                    f.reshape(-1, self.n_features),
                    target.reshape(-1, self.n_actions),
                    epochs=self.n_epochs,
                    # steps_per_epoch=100,
                    verbose=0,
                )

    def predict(self, features):
        y = self.model.predict(features.reshape(-1, self.n_features))
        # print(y)
        return y


class Forest(Regressor):
    def __init__(self, n_features, n_actions=6, n_estimators=10, max_depth=5, random_state=0):
        super().__init__(n_features, n_actions)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
        )

        # xas = [np.zeros(n_features + 1)]  # gamestate and action as argument
        # ys = [0]  # target response
        # self.forest.fit(xas, ys)

    def fit(self, features, values):
        """Use new data to update model. 'features' are vectors with gamestate features AND action as last element."""
        self.forest.fit(features, values)

    def predict(self, features):
        # returns = np.zeros(self.n_actions)
        Xs = np.tile(features, (6, 1))
        a = np.reshape(np.arange(6), (6, 1))
        xa = np.concatenate((Xs, a), axis=1)
        returns = self.forest.predict(xa)
        return returns



class LVA(Regressor):
    def __init__(self, n_features, n_actions=6, alpha=0.02):
        super().__init__(n_features, n_actions)
        self.beta = np.zeros((n_features, n_actions))
        self.alpha = alpha
        self.forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        # xas = [np.zeros(n_features + 1)]  # gamestate and action as argument
        # ys = [0]  # target response
        # self.forest.fit(xas, ys)

    def fit(self, features, values):
        # optimize Q towards Y
        for i in range(len(features)):
            state = np.array(features[i][:-1])
            action = features[i][-1]
            self.beta[:, action] += (
                ALPHA
                / len(len(features))
                * state
                * (values[i] - state.T @ self.beta[:, action])
            )

    def predict(self, features):
        return features @ self.beta

class GradientBoostingForest(Regressor):
    def __init__(self, n_features, n_actions=6, random_state=0, base=None, first_weight=0.1, mu=0.5):
        # TODO: think about if setting random state makes sense for us
        super().__init__(n_features, n_actions)

        self.random_state = random_state

        if base is None:
            self.forest = []
            self.weights = []
        else:
            self.forest = [base]
            self.weights = [1]
        self.first_weight = first_weight
        self.mu = mu

    def fit(self, features, values):
        # calculate residuals rho with prediction of old model
        rho = values - self.predict_vec_single_action(features)
        # fit decision stub on residuals of batch
        stub = DecisionTreeRegressor(max_depth=1, random_state=self.random_state)
        stub.fit(features, rho)

        # add decision stub to ensemble
        self.forest.append(stub)
        self.weights.append(self.first_weight / (1 + self.mu * len(self.forest)))

    def predict_vec_single_action(self, feature_vec_with_action):
        response = np.zeros(len(feature_vec_with_action))
        for i, features in enumerate(feature_vec_with_action):
            response[i] = self.predict(features[:-1])[features[-1]]
        return response

    def predict_vec(self, feature_vec):
        # set up response vector
        response = np.zeros((len(feature_vec), self.n_actions))  # one for each action
        # print("response.shape", response.shape)
        # combine features and actions to evaluate them separately -> xa
        #print("feature_vec.shape", feature_vec.shape)
        #feature_vec = np.reshape(feature_vec, (-1, 9))
        #print("feature_vec.shape", feature_vec.shape)
        Xs = np.tile(feature_vec, (self.n_actions, 1, 1))
        # print("Xs.shape", Xs.shape)
        a = np.reshape(np.arange(self.n_actions), (self.n_actions, 1))
        b = np.transpose(np.tile(a, (len(feature_vec), 1, 1)), (1, 0, 2))
        # print("a.shape", a.shape)
        # print("b.shape", b.shape)

        xa = np.reshape(np.concatenate((Xs, b), axis=2), (-1, 10))
        # print("xa.shape", xa.shape)

        # for each decision stub in ensemble
        for i, f in enumerate(self.forest):
            # predict response for each possible action in parallel and add it to total response with proper weight
            p = f.predict(xa)
            p_vec = np.reshape(p, (-1, 6))
            response += self.weights[i] * p_vec

        return response


    def predict(self, features):
        # set up response vector
        response = np.zeros(self.n_actions)  # one for each action

        # combine features and actions to evaluate them separately -> xa
        # print("feat.shape", features.shape)
        features = np.reshape(features, (-1, 9))
        # print("feat.shape", features.shape)
        Xs = np.tile(features, (self.n_actions, 1))
        # print("Xs.shape", Xs.shape)
        a = np.reshape(np.arange(self.n_actions), (self.n_actions, 1))
        xa = np.concatenate((Xs, a), axis=1)
        # print("xa.shape", xa.shape)

        # for each decision stub in ensemble
        for i, f in enumerate(self.forest):
            # predict response for each possible action in parallel and add it to total response with proper weight
            response += self.weights[i] * f.predict(xa)

        return response


class QMatrix:
    def __init__(self, Q):
        self.Q = Q

    def predict(self, features):
        #print(np.array(features).shape)
        extractions = np.concatenate(([0], np.arange(len(features), 0, -1)))

        # print(extractions.shape, extractions)
        # print(self.Q.shape)
        # print(features.shape)
        #return self.Q.transpose(extractions)[features]
        return self.Q[tuple(features.T)]
