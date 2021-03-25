# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:09:57 2021

@author: Philipp
"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.backend import clear_session

from sklearn.ensemble import RandomForestRegressor

import numpy as np


class Regressor:
    def __init__(self, n_features, n_actions=6):
        self.n_features = n_features
        self.n_actions = n_actions

    def fit(self, features, values):
        raise NotImplementedError("subclasses must override fit()!")

    def predict(self, features):
        raise NotImplementedError("subclasses must override predict()!")

    def update(self, transitions):
        raise NotImplementedError("subclasses must override update()!")


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
    def __init__(
        self, n_features, n_actions=6, n_estimators=10, max_depth=5, random_state=0
    ):
        super().__init__(n_features, n_actions)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        # xas = [np.zeros(n_features + 1)]  # gamestate and action as argument
        # ys = [0]  # target response
        # self.forest.fit(xas, ys)

    def fit(self, features, values):
        self.forest.fit(features, values)

    def predict(self, features):
        # returns = np.zeros(self.n_actions)
        Xs = np.tile(features, (6, 1))
        a = np.reshape(np.arange(6), (6, 1))
        xa = np.concatenate((Xs, a), axis=1)
        returns = self.forest.predict(xa)
        return returns
