# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:09:57 2021

@author: Philipp
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from sklearn.ensemble import RandomForestRegressor

from.callbacks import ACTIONS, state_to_features

import numpy as np

N = 2  # for n-step TD Q learning
GAMMA = 0.9
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

def get_all_rotations(index_vector):
    rots = np.reshape(
        index_vector, (1, -1)
    )  # makes list that only contains index_vector
    # would be cool to find a more readable numpy way...
    flipped_vector = flip(index_vector)  # check if already symmetric
    if flipped_vector not in rots:
        rots.append(flipped_vector)

    for i in range(0, 3):
        index_vector = rotate(index_vector)
        if index_vector not in rots:
            rots.append(index_vector)
        flipped_vector = flip(index_vector)
        if flipped_vector not in rots:
            rots.append(flipped_vector)
    return rots


def rotate(index_vector):
    """
    Rotates the state vector 90 degrees clockwise.
    """
    # TODO: BETTER FEATURES
    # feat = index_vector[:-1]
    # feat = np.reshape(feat, (7,7))
    # visual_feedback = False
    # if visual_feedback:
    #     visualize(feat, index_vector[-1])

    # rot = np.rot90(feat, k=1)

    if index_vector[-1] <= 3:  # DIRECTIONAL ACTION -> add 1
        action_index = (index_vector[-1] + 1) % 4
    else:
        action_index = index_vector[-1]  # BOMB and WAIT invariant

    rot = (
        index_vector[3],  # save tiles
        index_vector[0],
        index_vector[1],
        index_vector[2],
        -index_vector[5] + 4,  # POI vector y->-x
        index_vector[4],  # x->y
        index_vector[6],  # POI type invariant
        index_vector[7],  # POI distance invariant
        action_index,
    )
    # if visual_feedback:
    #     visualize(rot, action_index)
    #     print("=================================================================================")

    return rot

    # return np.concatenate((np.reshape(rot, (-1)), [action_index]))


def flip(index_vector):
    """
    Flips the state vector left to right.
    """
    # feat = index_vector[:-1]
    # feat = np.reshape(feat, (7, 7))
    # TODO: BETTER FEATURES
    # visual_feedback = False
    # if visual_feedback:
    #    visualize(feat, index_vector[-1])
    # flip = np.flipud(feat)      # our left right is their up down (coords are switched), check with visual feedback if you don't believe it ;)
    if index_vector[-1] == 1:  # LEFT RIGHT-> switch
        action_index = 3
    elif index_vector[-1] == 3:
        action_index = 1
    else:
        action_index = index_vector[-1]  # UP, DOWN, BOMB and WAIT invariant

    flip = (
        index_vector[0],  # surrounding
        index_vector[3],
        index_vector[2],
        index_vector[1],
        -index_vector[4] + 4,  # POI vector x->-x
        index_vector[5],  # y->y
        index_vector[6],  # POI type invariant
        index_vector[7],  # POI distance invariant
        action_index,
    )
    # if visual_feedback:
    #     visualize(flip, action_index)
    #     print("=================================================================================")
    return flip
    # return np.concatenate((np.reshape(flip, (-1)), [action_index]))

class Regressor:
    
    def __init__(self, n_features, n_actions=6):
        self.n_features = n_features
        self.n_actions = n_actions
    
    def fit(self, features, values):
        raise NotImplementedError('subclasses must override fit()!')
        
    def predict(self, features):
        raise NotImplementedError('subclasses must override predict()!')
        
    def update(self, transitions):
        raise NotImplementedError('subclasses must override update()!')
        

class Network(Regressor):
    
    def __init__(self, n_features, n_actions=6, n_layers=2, n_epochs=150):
        super().__init__(n_features, n_actions)
    
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        
        self.model = Sequential()

        #https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
        #create model
        #relu passt?
        self.model.add(Dense(n_actions, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
        for i in range(n_layers - 1):
            self.model.add(Dense(n_actions, activation='relu', kernel_initializer='he_normal'))
            
            #configures model for training
            self.model.compile(optimizer='adam', loss='mse')
            

    def fit(self, features, values):
        self.model.fit(features, values, epochs=self.n_epochs, verbose=0)
    
    
    def predict(self, features):
        return self.model.predict(features)
    
    

class Forest(Regressor):
    
    def __init__(self, n_features, n_actions=6, n_estimators=10, max_depth=5, random_state=0):
        super().__init__(n_features, n_actions)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.forest = RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state
        )
        
        xas = [np.zeros(n_features+1)]  # gamestate and action as argument
        ys = [0]  # target response
        self.forest.fit(xas, ys)
     
        
    def fit(self, features, values):
        self.forest.fit(features, values)
    
    
    def predict(self, features):
        #returns = np.zeros(self.n_actions)
        Xs = np.tile(features, (6, 1))
        a = np.reshape(np.arange(6), (6, 1))
        xa = np.concatenate((Xs, a), axis=1)
        returns = self.forest.predict(xa)
        return returns
    
    
    def update(self, transitions):
        batch = []
        occasion = []  # storing transitions in occasions to reflect their context
        for t in self.transitions:
            if t.state is not None:
                occasion.append(t)  # TODO: prioritize interesting transitions
            else:
                batch.append(occasion)
                occasion = []
                
        ys = []
        xas = []
        for occ in batch:
            for i, t in enumerate(occ):
                all_feat_action = get_all_rotations(
                    np.concatenate([occ[i].state, [ACTIONS.index(occ[i].action)]])
                    )
                for j in range(len(all_feat_action)):
                    # calculate target response Y using n step TD!
                    n = min(
                        len(occ) - i, N
                        )  # calculate next N steps, otherwise just as far as possible
                    r = [GAMMA ** k * occ[i + k].reward for k in range(n)]
                    # TODO: Different Y models
                    if t.next_state is not None:
                        Y = sum(r) + GAMMA ** n * np.max(self.predict(state_to_features(t.next_state)))
                    else:
                            Y = t.reward
                            ys.append(Y)
                            xas.append(all_feat_action[j])
                            xas = np.array(xas)
                            ys = np.array(ys)
                            self.regressor.fit(xas, ys)