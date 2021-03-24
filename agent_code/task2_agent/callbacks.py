import os
import pickle
import random
import numpy as np
from random import shuffle

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
EPSILON = 0.025


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

    ### CHANGE ONLY IF YOU KNOW WHAT YOU ARE DOING, no sync with training!
    self.model_suffix = ""

    # TEST BENCH CODE
    if "TESTING" in os.environ:
        if os.environ["TESTING"] == "YES":
            self.test_results = {"crates": [], "total_crates": []}
            self.model_suffix = "_" + os.environ["MODELNAME"]
            self.total_crates = 0
            self.last_crates = 0
            print("WARNING: TESTING (perhaps on a different model!)")
    if self.train or not os.path.isfile(f"model{self.model_suffix}/model.pt"):
        self.logger.info("Setting up model from scratch.")

    else:
        self.logger.info("Loading model.")
        print("Loading model.")
        with open(f"model{self.model_suffix}/model.pt", "rb") as file:
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

    # TEST BENCH CODE
    if "TESTING" in os.environ:
        if os.environ["TESTING"] == "YES":
            crates = np.count_nonzero(game_state["field"] == 1)
            if self.total_crates == 0 or self.last_crates < crates:
                self.total_crates = crates
                self.test_results["total_crates"].append(crates)
                self.test_results["crates"].append(crates)
            elif self.last_crates > crates:
                self.test_results["crates"][-1] = crates
            self.last_crates = crates

            with open(f"model{self.model_suffix}/test_results.pt", "wb") as file:
                pickle.dump(self.test_results, file)

    # ->EPSILON greedy

    if self.train and random.random() < EPSILON:
        self.logger.debug("EPSILON-greedy: Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    # start = time.time()
    Qs = Q(self, feat)
    self.logger.debug("Qs for this situation: " + str(Qs))
    action_index = np.random.choice(np.flatnonzero(Qs == np.max(Qs)))
    # self.logger.debug("Choosing an action took " + str((time.time() - start)) + "ms.")

    # -> soft-max

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

    # BETTER FEATURES
    # sight = 3
    # field = game_state["field"]
    #
    # for coord in game_state["coins"]:
    #     field[coord] = 2
    # for coord, tick in game_state["bombs"]:
    #     field[coord] = tick + 3     # assign numbers greater than 3 to bombs, simultaneously coding the ticks
    #
    # field_size = len(field)
    # greater_field = -np.zeros((field_size+(sight-1)*2, field_size+(sight-1)*2))
    # greater_field[sight-1:-(sight-1), sight-1:-(sight-1)] = field
    #
    # x, y = game_state["self"][3]
    # x += sight - 1
    # y += sight - 1
    # visible_field = greater_field[x-sight:x+sight+1, y-sight:y+sight+1]
    #
    # visual_feedback = False
    # if visual_feedback:
    #    for j in range(2*sight+1):
    #         s = "|"
    #         for i in range(2*sight+1):
    #             if visible_field[i, j] == -1:
    #                 s += "XXX"
    #             elif visible_field[i, j] == 1:
    #                 s += "(-)"
    #             elif visible_field[i, j] == 0:
    #                 s += "   "
    #             elif visible_field[i, j] == 2:
    #                 s += " $ "
    #             elif visible_field[i, j] >= 3:
    #                 s += "!%i!" % (visible_field[i, j]-3)
    #             else:
    #                 raise
    #         print(s+"|")
    #
    # return np.reshape(visible_field, (-1))

    # mod_pos = [game_state["self"][3][0] % 2, game_state["self"][3][1] % 2]

    x_off = [0, 1, 0, -1]
    y_off = [-1, 0, 1, 0]

    # part for computing POI and save

    # for bombs:
    if game_state["bombs"] != []:
        # TODO: schlechter fix for now, immer nur eine Bombe(die eigene) in Task 2
        # deswegen:
        POI_position = game_state["bombs"][0][0]
        POI_type = 0

        free_space = game_state["field"] == 0
        free_space[tuple(game_state["bombs"][0][0])] = False

        start = game_state["self"][3]

        save = [0, 0, 0, 0]
        x, y = start

        dist = game_state["bombs"][0][0] - np.array(start)

        # if tile already save, show surrounding tiles as [0,0,0,0] (should WAIT)
        if not (all(dist != 0) or np.sum(np.abs(dist)) > 3):
            # print("current position not save!")
            # else: search if tiles are save
            neighbors = [
                (x, y)
                for (x, y) in [
                    (x + x_off[0], y + y_off[0]),
                    (x + x_off[1], y + y_off[1]),
                    (x + x_off[2], y + y_off[2]),
                    (x + x_off[3], y + y_off[3]),
                ]
            ]

            for i, neighbor in enumerate(neighbors):
                if free_space[neighbor]:
                    # print("checking..", neighbor)
                    dist = game_state["bombs"][0][0] - np.array(neighbor)
                    if all(dist != 0) or np.sum(np.abs(dist)) > 3:
                        # print("neighbor is save!", neighbor)
                        save[i] = 1
                        continue
                    frontier = [neighbor]
                    parent_dict = {start: start, neighbor: neighbor}
                    dist_so_far = {neighbor: 1}
                    while len(frontier) > 0:
                        current = frontier.pop(0)
                        if dist_so_far[current] > game_state["bombs"][0][1] + 1:
                            # print("too far: stopping here", current)
                            continue
                        x, y = current
                        available_neighbors = [
                            (x, y)
                            for (x, y) in [
                                (x + 1, y),
                                (x - 1, y),
                                (x, y + 1),
                                (x, y - 1),
                            ]
                            if free_space[x, y]
                        ]

                        for neineighbor in available_neighbors:
                            if neineighbor not in parent_dict:
                                frontier.append(neineighbor)
                                parent_dict[neineighbor] = neighbor
                                dist = game_state["bombs"][0][0] - np.array(neineighbor)
                                dist_so_far[neineighbor] = dist_so_far[current] + 1
                                if all(dist != 0) or np.sum(np.abs(dist)) > 3:
                                    save[i] = 1
                                    # print("found save spot at ", neineighbor)
                                    break
                        else:
                            continue
                        break
        # print(save)
    else:
        # for crates, coins: BFS

        start = game_state["self"][3]
        frontier = [start]
        parent_dict = {start: start}
        dist_so_far = {start: 0}
        found_targets = []  # list of tuples (*coord, type)

        free_space = game_state["field"] == 0

        while len(frontier) > 0:
            current = frontier.pop(0)
            if current in game_state["coins"]:
                found_targets.append([current, 2, dist_so_far[current]])

            # Add unexplored free neighboring tiles to the queue in a random order
            x, y = current
            neighbors = [
                (x, y)
                for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
                if free_space[x, y]
            ]
            all_neighbors = [
                (x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            ]
            shuffle(all_neighbors)
            for neighbor in all_neighbors:
                if game_state["field"][neighbor] == 1:  # CRATE
                    found_targets.append([neighbor, 1, dist_so_far[current] + 1])

            shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in parent_dict:
                    frontier.append(neighbor)
                    parent_dict[neighbor] = current
                    dist_so_far[neighbor] = dist_so_far[current] + 1

        # print(found_targets)
        if len(found_targets) == 0:
            POI_position = game_state["self"][3]
            POI_type = 1
            POI_dist = 0
        else:
            found = sorted(found_targets, key=lambda tar: tar[2])[0]
            # print(found)
            # try:
            #     found_ind = np.argmin(np.array(found_targets, dtype=object)[:, 2], axis=0)
            # except:
            #     print("WTF, encountered weird found_targets: " + str(found_targets))
            #     found_ind = 0
            # found = found_targets[found_ind]
            POI_position = found[0]
            POI_type = found[1]

        # ALSO compute save directions
        save = np.zeros(len(x_off))
        for i in range(len(save)):
            save[i] = (
                -np.abs(
                    game_state["field"][
                        game_state["self"][3][0] + x_off[i],
                        game_state["self"][3][1] + y_off[i],
                    ]
                )
                + 1
            )

    dist = POI_position - np.array(game_state["self"][3])
    bigger = np.argmax(np.abs(dist))
    POI_vector = np.sign(dist) + 1
    POI_vector[bigger] *= 2
    POI_dist = np.clip(np.sum(np.abs(dist)), a_max=4, a_min=0)

    # channels = []
    # channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)
    # and return them as a vector
    return np.concatenate((save, POI_vector, [POI_type], [POI_dist])).astype(int)
    # stacked_channels.reshape(-1)


#### UTILITY FUNCTIONS


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
    # else: raise

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


def visualize_old(index_vector):
    # x heißt nicht safe, o heißt safe
    # safe_chars = ["x" if (index_vector[i] == 0) else "o" for i in range(4)]
    s = ""
    # l heißt left, r right, u up, d down
    if index_vector[4] == 0:
        s += "l"
    if index_vector[4] == 2:
        s += "r"
    if index_vector[5] == 0:
        s += "u"
    if index_vector[5] == 2:
        s += "d"
    # ? für die felder die nicht gecheckt werden
    # print("This visualizes to")
    # print("? ", safe_chars[0], "  ?")
    # print(safe_chars[3], " ", s, " ", safe_chars[1])
    # print("? ", safe_chars[2], "  ?")

    # print("With action ", ACTIONS[index_vector[8]])
