import os
import pickle
import random

import numpy as np


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT"]  # , "BOMB"]


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("model.pt"):
        self.logger.info("Setting up model from scratch.")

    else:
        self.logger.info("Loading model.")
        with open("model.pt", "rb") as file:
            self.Q = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    feat = state_to_features(game_state)
    self.logger.debug("Querying model for action with feature " + str(feat))

    # epsilon greedy
    epsilon = 0.05
    if self.train and random.random() < epsilon:
        self.logger.debug("Epsilon-greedy: Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    action_index = np.argmax(
        self.Q[:, int(feat[0] + 14), int(feat[1] + 14), int(feat[2]), int(feat[3])]
    )

    # soft-max
    # ROUNDS = 100000
    # rho = np.clip((1 - game_state["round"]/ROUNDS)*0.7, a_min=1e-3, a_max=0.5) # starte sehr kalt, wegen gutem anfangsQ
    # Qvals = self.Q[:, int(feat[0] + 14), int(feat[1] + 14), int(feat[2]), int(feat[3])]
    # softmax = np.exp(Qvals/rho)/np.sum(np.exp(Qvals/rho))
    # self.logger.debug("softmax:" + str(softmax))
    # action_index = np.random.choice(np.arange(len(ACTIONS)), p=softmax)

    self.logger.debug("ACTION choosen: " + ACTIONS[action_index])
    return ACTIONS[action_index]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    if game_state["coins"] != []:
        distance = game_state["coins"] - np.array(game_state["self"][3])
        closest_index = np.argmin(np.sum(np.abs(distance), axis=1))
        closest_vector = distance[closest_index]
    else:
        closest_vector = [0, 0]  # treat non-existing coins as [0,0]

    # check if surrounding tiles are blocked
    # x_off = [1, -1, 0, 0]
    # y_off = [0, 0, 1, -1]
    # blocked = np.zeros(4)
    # for i in range(len(blocked)):
    #    blocked[i] = game_state["field"][
    #        game_state["self"][3][0] + x_off[i], game_state["self"][3][1] + y_off[i]
    #    ]

    mod_pos = [game_state["self"][3][0] % 2, game_state["self"][3][1] % 2]

    # channels = []
    # channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)
    # and return them as a vector
    return np.concatenate((closest_vector, mod_pos))  # stacked_channels.reshape(-1)
