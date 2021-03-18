import os
import pickle
import random

import numpy as np


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


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
    self.logger.debug("Querying model for action with feature " + str(tuple(feat)))
    # TODO: Exploration vs exploitation

    # epsilon greedy
    epsilon = 0.05
    if self.train and random.random() < epsilon:
        self.logger.debug("Epsilon-greedy: Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    # get all symmetries
    Qs = []
    for act in range(len(ACTIONS)):
        origin_vec = np.concatenate((feat, [act]))
        # encountered_symmetry = False
        for rot in get_all_rotations(origin_vec):
            if self.Q.already_exists(rot):
                # encountered_symmetry = True
                Qs.append(self.Q.get_entry(rot))
                break
        else:
            Qs.append(0)
        # if not encountered_symmetry:

    action_index = np.random.choice(np.flatnonzero(Qs == np.max(Qs)))

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
    # WIRD IM MOMENT 3 MAL AUSGEFÜHRT

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    # TODO: channels?

    if game_state["bombs"] != []:
        # bombs_dist = np.matrix(game_state["bombs"][:, 0])
        # print(bombs_dist)
        # distance = bombs_dist - np.array(game_state["self"][3])
        # closest_index = np.argmin(
        #     np.sum(np.abs(distance), axis=1)
        # )  # manhattan distance
        # closest_bomb = distance[closest_index] + 14
        # TODO: schlechter fix for now, immer nur eine Bombe(die eigene) in Task 2
        # deswegen:
        closest_bomb = (
            game_state["bombs"][0][0] - np.array(game_state["self"][3]) + np.full(2, 14)
        )
        bomb_ticker = game_state["bombs"][0][1]
    else:
        closest_bomb = [14, 14]  # treat non-existing bombs as [0,0]
        bomb_ticker = 3

    # For example, you could construct several channels of equal shape, ...
    # if game_state["coins"] != []:
    #     distance = game_state["coins"] - np.array(game_state["self"][3])
    #     closest_index = np.argmin(np.sum(np.abs(distance), axis=1))
    #     closest_coin = distance[closest_index]
    # else:
    #     closest_coin = [0, 0]  # treat non-existing coins as [0,0]

    # check surrounding tiles
    # x_off = [0, 1, 1, 1, 0, -1, -1, -1]  # , 2, -2, 0, 0]
    # y_off = [1, 1, 0, -1, -1, -1, 0, 1]  # , 0, 0, 2, -2]
    x_off = [0, 1, 0, -1]  # , 2, -2, 0, 0]
    y_off = [1, 0, -1, 0]  # , 0, 0, 2, -2]
    around_me = np.zeros(len(x_off))
    for i in range(len(around_me)):
        if (
            game_state["self"][3][0] + x_off[i] > 16
            or game_state["self"][3][0] + x_off[i] < 0
        ):
            around_me[i] = 0
        elif (
            game_state["self"][3][1] + y_off[i] > 16
            or game_state["self"][3][1] + y_off[i] < 0
        ):
            around_me[i] = 0
        else:
            around_me[i] = (
                game_state["field"][
                    game_state["self"][3][0] + x_off[i],
                    game_state["self"][3][1] + y_off[i],
                ]
                + 1
            )

    # mod_pos = [game_state["self"][3][0]%2, game_state["self"][3][1]%2]

    # channels = []
    # channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)
    # and return them as a vector
    return np.concatenate((closest_bomb, [bomb_ticker], around_me)).astype(int)
    # stacked_channels.reshape(-1)


#### UTILITY FUNCTIONS


def get_all_rotations(index_vector):
    rots = [index_vector, flip(index_vector)]
    for i in range(0, 3):
        index_vector = rotate(index_vector)
        rots.append(index_vector)
        index_vector = flip(index_vector)
        rots.append(index_vector)
    return rots


def rotate(index_vector):
    """
    Rotates the state vector 90 degrees clockwise.
    """
    # 11
    if index_vector[7] <= 3:  # DIRECTIONAL ACTION -> add 1
        action_index = (index_vector[7] + 1) % 4
    else:
        action_index = index_vector[7]  # BOMB and WAIT invariant

    return (
        -index_vector[1] + 28,  # bomb position y->-x
        index_vector[0],  # x->y
        index_vector[2],  # bomb ticker invariant
        # index_vector[6 + 3],  # surrounding
        # index_vector[7 + 3],
        # index_vector[0 + 3],
        # index_vector[1 + 3],
        # index_vector[2 + 3],
        # index_vector[3 + 3],
        # index_vector[4 + 3],
        # index_vector[5 + 3],
        index_vector[3 + 3],  # surrounding
        index_vector[0 + 3],
        index_vector[1 + 3],
        index_vector[2 + 3],
        action_index,
    )


def flip(index_vector):
    """
    Flips the state vector left to right.
    """

    if index_vector[7] == 1:  # DIRECTIONAL ACTION -> add 1
        action_index = 3
    elif index_vector[7] == 3:
        action_index = 1
    else:
        action_index = index_vector[7]  # UP, DOWN, BOMB and WAIT invariant

    return (
        -index_vector[0] + 28,  # bomb position x->-x
        index_vector[1],  # y->y
        index_vector[2],  # bomb ticker invariant
        # index_vector[0 + 3],  # surrounding
        # index_vector[7 + 3],
        # index_vector[6 + 3],
        # index_vector[5 + 3],
        # index_vector[4 + 3],
        # index_vector[3 + 3],
        # index_vector[2 + 3],
        # index_vector[1 + 3],
        index_vector[0 + 3],  # surrounding
        index_vector[3 + 3],
        index_vector[2 + 3],
        index_vector[1 + 3],
        action_index,
    )
