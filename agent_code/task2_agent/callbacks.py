import os
import pickle
import random
import time
import numpy as np
from random import shuffle

from sklearn.ensemble import RandomForestRegressor

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
EPSILON = 0.1


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
    if not os.path.isfile("model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.forest = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=0)
        xas = [np.zeros(50)] # gamestate and action as argument
        ys = [0]            # target response
        self.forest.fit(xas, ys)

    else:
        self.logger.info("Loading model.")
        with open("model.pt", "rb") as file:
            self.forest = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # print("-----------------")
    feat = state_to_features(game_state)
    self.logger.debug(
        "Querying model for action with feature " + str(tuple(feat)) + "."
    )
    # TODO: Exploration vs exploitation

    # EPSILON greedy

    if self.train and random.random() < EPSILON:
        self.logger.debug("EPSILON-greedy: Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    # start = time.time()
    # get all symmetries
    # Qs = []
    # for act in range(len(ACTIONS)):
    #     origin_vec = np.concatenate((feat, [act]))
    #     # encountered_symmetry = False
    #     for rot in get_all_rotations(origin_vec):
    #         if self.Q.already_exists(rot):
    #             # encountered_symmetry = True
    #             Qs.append(self.Q.get_entry(rot))
    #             break
    #     else:  # if not encountered symmetry
    #         Qs.append(0)
    Qs = Q(self, feat)

    action_index = np.random.choice(np.flatnonzero(Qs == np.max(Qs)))
    # self.logger.debug("Choosing an action took " + str((time.time() - start)) + "ms.")

    # soft-max

    # ROUNDS = 100000
    # rho = np.clip((1 - game_state["round"]/ROUNDS)*0.7, a_min=1e-3, a_max=0.5) # starte sehr kalt, wegen gutem anfangsQ
    # Qvals = self.Q[:, int(feat[0] + 14), int(feat[1] + 14), int(feat[2]), int(feat[3])]
    # softmax = np.exp(Qvals/rho)/np.sum(np.exp(Qvals/rho))
    # self.logger.debug("softmax:" + str(softmax))
    # action_index = np.random.choice(np.arange(len(ACTIONS)), p=softmax)

    # print(feat)
    # print("ACTION choosen: " + ACTIONS[action_index])
    self.logger.debug("ACTION choosen: " + ACTIONS[action_index])
    return ACTIONS[action_index]


def Q(self, X) -> np.array:
    Xs = np.tile(X, (6, 1))
    a = np.reshape(np.arange(6), (6, 1))
    xa = np.concatenate((Xs, a), axis=1)
    return self.forest.predict(xa)


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

    sight = 3
    field = game_state["field"]

    for coord in game_state["coins"]:
        field[coord] = 2
    for coord, tick in game_state["bombs"]:
        field[coord] = tick + 3     # assign numbers greater than 3 to bombs, simultaneously coding the ticks

    field_size = len(field)
    greater_field = -np.zeros((field_size+(sight-1)*2, field_size+(sight-1)*2))
    greater_field[sight-1:-(sight-1), sight-1:-(sight-1)] = field

    x, y = game_state["self"][3]
    x += sight - 1
    y += sight - 1
    visible_field = greater_field[x-sight:x+sight+1, y-sight:y+sight+1]

    visual_feedback = False
    if visual_feedback:
       for j in range(2*sight+1):
            s = "|"
            for i in range(2*sight+1):
                if visible_field[i, j] == -1:
                    s += "XXX"
                elif visible_field[i, j] == 1:
                    s += "(-)"
                elif visible_field[i, j] == 0:
                    s += "   "
                elif visible_field[i, j] == 2:
                    s += " $ "
                elif visible_field[i, j] >= 3:
                    s += "!%i!" % (visible_field[i, j]-3)
                else:
                    raise
            print(s+"|")

    return np.reshape(visible_field, (-1))


#### UTILITY FUNCTIONS


def get_all_rotations(index_vector):
    rots = np.reshape(index_vector, (1,-1))     # makes list that only contains index_vector
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

    feat = index_vector[:-1]
    feat = np.reshape(feat, (7,7))

    visual_feedback = False
    if visual_feedback:
        visualize(feat, index_vector[-1])

    rot = np.rot90(feat, k=1)

    if index_vector[-1] <= 3:  # DIRECTIONAL ACTION -> add 1
        action_index = (index_vector[-1] + 1) % 4
    else:
        action_index = index_vector[-1]  # BOMB and WAIT invariant

    if visual_feedback:
        visualize(rot, action_index)
        print("=================================================================================")

    return np.concatenate((np.reshape(rot, (-1)), [action_index]))


def flip(index_vector):
    """
    Flips the state vector left to right.
    """
    feat = index_vector[:-1]
    feat = np.reshape(feat, (7, 7))

    visual_feedback = False
    if visual_feedback:
        visualize(feat, index_vector[-1])
    flip = np.flipud(feat)      # our left right is their up down (coords are switched), check with visual feedback if you don't believe it ;)
    if index_vector[-1] == 1:  # LEFT RIGHT-> switch
        action_index = 3
    elif index_vector[-1] == 3:
        action_index = 1
    else:
        action_index = index_vector[-1]  # UP, DOWN, BOMB and WAIT invariant
    if visual_feedback:
        visualize(flip, action_index)
        print("=================================================================================")

    return np.concatenate((np.reshape(flip, (-1)), [action_index]))


def visualize(feat, action_index):
    print("The resulting vector is: ")
    sight = 3
    print("action:", end="")
    if action_index == 0:
        print("↑")
    elif action_index == 1:
        print("→")
    elif action_index == 2:
        print("↓")
    elif action_index == 3:
        print("←")
    elif action_index == 4:
        print("⊗")
    elif action_index == 5:
        print("💣")
    else: raise

    for j in range(2 * sight + 1):
        s = "|"
        for i in range(2 * sight + 1):
            if feat[i, j] == -1:
                s += "XXX"
            elif feat[i, j] == 1:
                s += "(-)"
            elif feat[i, j] == 0:
                s += "   "
            elif feat[i, j] == 2:
                s += " $ "
            elif feat[i, j] >= 3:
                s += "!%i!" % (feat[i, j] - 3)
            else:
                raise
        print(s + "|")


