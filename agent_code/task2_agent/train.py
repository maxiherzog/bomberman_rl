import pickle
from collections import namedtuple, deque
from typing import List
import os

import json

import events as e
from .callbacks import state_to_features
from .callbacks import ACTIONS
from .callbacks import get_all_rotations
from .callbacks import EPSILON_MAX
from .callbacks import EPSILON_MIN
from .callbacks import EPSILON_DECAY
import numpy as np
from .regressors import Forest
from .regressors import GradientBoostingForest
from .regressors import QMatrix

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Events
EVADED_BOMB = "EVADED_BOMB"
NO_CRATE_DESTROYED = "NO_CRATE_DESTROYED"
NO_BOMB = "NO_BOMB"
BLOCKED_SELF_IN_UNSAFE_SPACE = "BLOCKED_SELF_IN_UNSAFE_SPACE"
DROPPED_BOMB_NEXT_TO_CRATE = "DROPPED_BOMB_NEXT_TO_CRATE"
NEW_PLACE = "NEW_PLACE"
NO_ACTIVE_BOMB = "NO_ACTIVE_BOMB"

# Hyper parameters -- DO modify
# TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
GAMMA = 0.93
ALPHA = 0.2
N = 1  # for n-step TD Q learning
XP_BUFFER_SIZE = 100  # higher batch size for forest
N_ESTIMATORS = 100
MAX_DEPTH = 40

EXPLOIT_SYMMETRY = True
GAME_REWARDS = {
    # HANS
    # e.KILLED_OPPONENT: 5,
    e.COIN_COLLECTED: 2,
    e.INVALID_ACTION: -0.1,
    e.CRATE_DESTROYED: 0.7,
    e.KILLED_SELF: -0.5,
    e.BOMB_DROPPED: 0.02,
    DROPPED_BOMB_NEXT_TO_CRATE: 0.2,
    EVADED_BOMB: 0.1,
    NO_CRATE_DESTROYED: -0.3,
    NO_ACTIVE_BOMB: -0.1,
    BLOCKED_SELF_IN_UNSAFE_SPACE: -0.3,
    # MAXI
    # e.COIN_COLLECTED: 1,
    # e.INVALID_ACTION: -1,
    # BLOCKED_SELF_IN_UNSAFE_SPACE: -10,
    # e.CRATE_DESTROYED: 0.1,
    # NO_BOMB: -0.05,
    # NO_CRATE_DESTROYED: -3
    # PHILIPP
    # e.COIN_COLLECTED: 5,
    # e.INVALID_ACTION: -0.5,
    # e.CRATE_DESTROYED: 3,
    # e.KILLED_SELF: -4,
    # e.BOMB_DROPPED: 0.05,
    # NO_BOMB: -0.02,
    # DROPPED_BOMB_NEXT_TO_CRATE: 0.4,
    # TÜFTEL
    # e.COIN_COLLECTED: 1,
    # # e.KILLED_OPPONENT: 5,
    # e.INVALID_ACTION: -1,
    # e.CRATE_DESTROYED: 0.5,
    # e.BOMB_DROPPED: 0.05,
    # EVADED_BOMB: 0.25,
    # NO_BOMB: -0.01,
    # BLOCKED_SELF_IN_UNSAFE_SPACE: -5,
    # e.KILLED_SELF: -2,
    # NO_CRATE_DESTROYED: -3,
}

STORE_FREQ = XP_BUFFER_SIZE


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=None)  #
    self.rounds_played = 0

    #self.regress = False
    # ensure model subfolder
    if not os.path.exists("model"):
        os.makedirs("model")

    if os.path.isfile("model/model.pt"):
        self.logger.info("Retraining from saved state.")

        self.logger.info("Reloading analysis variables.")
        with open("model/analysis_data.pt", "rb") as file:
            self.analysis_data = pickle.load(file)

    else:
        if "AUTOTRAIN" in os.environ:
            if os.environ["AUTOTRAIN"] == "YES":
                global ALPHA, GAMMA
                ALPHA = float(os.environ["ALPHA"])
                GAMMA = float(os.environ["GAMMA"])

        self.logger.debug("Initializing Q")
        Q = np.zeros([2, 2, 2, 2, 5, 5, 2, 2, 5, len(ACTIONS)])

        # dont run into walls
        Q[0, :, :, :, :, :, :, :, :, 0] += -2
        Q[:, 0, :, :, :, :, :, :, :, 1] += -2
        Q[:, :, 0, :, :, :, :, :, :, 2] += -2
        Q[:, :, :, 0, :, :, :, :, :, 3] += -2

        # drop Bomb when near crate
        Q[:, :, :, :, :, :, 1, 0, 1, 5] += 2
        # dont drop Bomb when already having one bomb
        Q[:, :, :, :, :, :, 0, 0, :, 5] += -2

        # dont drop bomb when not near crate
        Q[:, :, :, :, :, :, 1, 0, 2:, 5] += -2
        # or near coin
        Q[:, :, :, :, :, :, 1, 1, :, 5] += -2
        # # walk towards crates
        # Q[1, :, :, :, :, :2, 1, 0, 2:, 0] += 1
        # Q[:, 1, :, :, -2:, :, 1, 0, 2:, 1] += 1
        # Q[:, :, 1, :, :, -2:, 1, 0, 2:, 2] += 1
        # Q[:, :, :, 1, :2, :, 1, 0, 2:, 3] += 1
        #
        # # walk towards coins
        # Q[1, :, :, :, :, :2, 1, 1, :, 0] += 1
        # Q[:, 1, :, :, -2:, :, 1, 1, :, 1] += 1
        # Q[:, :, 1, :, :, -2:, 1, 1, :, 2] += 1
        # Q[:, :, :, 1, :2, :, 1, 1, :, 3] += 1

        # walk away from bomb (only if safe) if ON BOMB
        Q[1, :, :, :, :, :, 0, 0, :, 0] += 1
        Q[:, 1, :, :, :, :, 0, 0, :, 1] += 1
        Q[:, :, 1, :, :, :, 0, 0, :, 2] += 1
        Q[:, :, :, 1, :, :, 0, 0, :, 3] += 1

        # and in straight lines
        # Q[:, :, 1, :, 1, 0, 0, 1:, 2] += 1
        # Q[:, :, :, 1, 2, 1, 0, 1:, 3] += 1
        # Q[1, :, :, :, 1, 2, 0, 1:, 0] += 1
        # Q[:, 1, :, :, 0, 1, 0, 1:, 1] += 1

        # and dont fucking WAIT
        Q[:, :, :, :, :, :, 0, 0, :, 4] += -2
        # but consider waiting if safe/dead
        Q[0, 0, 0, 0, :, :, 0, 0, :, 4] += 2

        xas = []  # gamestate and action as argument
        ys = []  # target response

        # set up prior matrix
        for i0 in range(Q.shape[0]):
            for i1 in range(Q.shape[1]):
                for i2 in range(Q.shape[2]):
                    for i3 in range(Q.shape[3]):
                        for i4 in range(Q.shape[4]):
                            for i5 in range(Q.shape[5]):
                                for i6 in range(Q.shape[6]):
                                    for i7 in range(Q.shape[7]):
                                        for i8 in range(Q.shape[8]):
                                            for a in range(Q.shape[8]):
                                                if (
                                                        Q[
                                                            i0,
                                                            i1,
                                                            i2,
                                                            i3,
                                                            i4,
                                                            i5,
                                                            i6,
                                                            i7,
                                                            i8,
                                                            a,
                                                        ]
                                                        != 0
                                                ):
                                                    xas.append(
                                                        [
                                                            i0,
                                                            i1,
                                                            i2,
                                                            i3,
                                                            i4,
                                                            i5,
                                                            i6,
                                                            i7,
                                                            i8,
                                                            a,
                                                        ]
                                                    )
                                                    ys.append(
                                                        Q[
                                                            i0,
                                                            i1,
                                                            i2,
                                                            i3,
                                                            i4,
                                                            i5,
                                                            i6,
                                                            i7,
                                                            i8,
                                                            a,
                                                        ]
                                                    )

        self.regress = False    # switch this here to decide which one you want to train from scratch!
        if self.regress:
            print("train a new regression based Forest")
            self.logger.info("train a new regression based Forest.")

            # prior = QMatrix(Q)
            # self.regressor = GradientBoostingForest(
            #     n_features=9, random_state=0, base=prior, first_weight=0.1, mu=0.5
            # )

            self.regressor = Forest(
                9, n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=0
            )
            xas = np.array(xas)
            ys = np.array(ys)
            self.regressor.fit(xas, ys)

        else:
            print("train a new non-regression matrix Q")
            self.logger.info("train a new non-regression matrix Q")
            self.Q = Q


        # init measured variables
        self.analysis_data = {
            # "Q_sum": [],
            # "Q_sum_move": [],
            # "Q_sum_bomb": [],
            # "Q_sum_wait": [],
            # "Q_situation": [self.Q[0, 1, 1, 0, 1, 2, 1, 2, :]],
            "reward": [],
            "win": [],
            "coins": [],
            "crates": [],
            "length": [],
            "bombs": [],
            "useless_bombs": [],
        }

        # dump hyper parameters as json
        hyperparams = {
            "GAMMA": GAMMA,
            "EPSILON_GREEDY": {
                "EPSILON_MAX": EPSILON_MAX,
                "EPSILON_MIN": EPSILON_MIN,
                "EPSILON_DECAY": EPSILON_DECAY,
            },
            "XP_BUFFER_SIZE": XP_BUFFER_SIZE,
            "N": N,
            "EXPLOIT_SYMMETRY": EXPLOIT_SYMMETRY,
            "REGRESSION TREE": {"N_ESTIMATORS": N_ESTIMATORS, "MAX_DEPTH": MAX_DEPTH, },
            "GAME_REWARDS": GAME_REWARDS,
        }
        with open("model/hyperparams.json", "w") as file:
            json.dump(hyperparams, file, ensure_ascii=False, indent=4)
        store(self)

    # init counters
    # hands on variables
    self.crate_counter = 0
    self.coin_counter = 0
    self.bombs_counter = 0
    self.useless_bombs_counter = 0


def game_events_occurred(
        self,
        old_game_state: dict,
        self_action: str,
        new_game_state: dict,
        events: List[str],
):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(
        f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}'
    )

    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)
    if e.BOMB_EXPLODED in events and not e.KILLED_SELF in events:
        events.append(EVADED_BOMB)

    if e.BOMB_EXPLODED in events and not e.CRATE_DESTROYED in events:
        events.append(NO_CRATE_DESTROYED)
        self.useless_bombs_counter += 1

    if not e.BOMB_DROPPED in events:
        events.append(NO_BOMB)

    if e.BOMB_DROPPED in events:
        self.bombs_counter += 1

    if e.CRATE_DESTROYED in events:
        self.crate_counter += 1

    if e.COIN_COLLECTED in events:
        self.coin_counter += 1

    old_feat = state_to_features(old_game_state)
    new_feat = state_to_features(new_game_state)
    # BLOCKED_SELF_IN_UNSAFE_SPACE
    if new_game_state["bombs"] != []:
        dist = new_game_state["bombs"][0][0] - np.array(new_game_state["self"][3])
        if all(new_feat[:4] == 0) and not (all(dist != 0) or np.sum(np.abs(dist)) > 3):
            events.append(BLOCKED_SELF_IN_UNSAFE_SPACE)
    else:
        events.append(NO_ACTIVE_BOMB)

    # if e.MOVED_UP or e.MOVED_DOWN or e.MOVED_LEFT or e.MOVED_RIGHT:
    #     if new_game_state["self"][0] not in self.visited:
    #         self.visited.append(new_game_state["self"][3])
    #         events.append(NEW_PLACE)

    # DROPPED_BOMB_NEXT_TO_CRATE
    if e.BOMB_DROPPED in events:
        if old_feat[6] == 1 and old_feat[7] == 1:
            events.append(DROPPED_BOMB_NEXT_TO_CRATE)

    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(old_feat, self_action, new_feat, reward_from_events(self, events))
    )


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :type self: object
    :param self: The same object that is passed to all of your callbacks.
    """

    self.epsilon = np.clip(
        self.epsilon * EPSILON_DECAY, a_max=EPSILON_MAX, a_min=EPSILON_MIN
    )
    self.logger.debug(
        f'Encountered event(s) {", ".join(map(repr, events))} in final step'
    )
    self.transitions.append(
        Transition(
            state_to_features(last_game_state),
            last_action,
            None,
            reward_from_events(self, events),
        )
    )

    self.analysis_data["win"].append(last_game_state["self"][1] == 9)
    self.analysis_data["crates"].append(self.crate_counter)
    self.analysis_data["coins"].append(self.coin_counter)
    self.analysis_data["length"].append(last_game_state["step"])
    self.analysis_data["bombs"].append(self.bombs_counter)
    self.analysis_data["useless_bombs"].append(self.useless_bombs_counter)

    # RESET Counters
    self.crate_counter = 0
    self.coin_counter = 0
    self.bombs_counter = 0
    self.useless_bombs_counter = 0

    self.rounds_played += 1
    if self.regress:
        if self.rounds_played % XP_BUFFER_SIZE == 0:
            updateQ(self)
            self.transitions = deque(maxlen=None)
    else:
        updateQMatrix(self)
        self.transitions = deque(maxlen=None)   # clear transitions -> ready for next game



    if last_game_state["round"] % STORE_FREQ == 0:
        store(self)


def store(self):
    """
    Stores all the files.
    """
    # Store the model
    self.logger.debug("Storing model.")
    with open(r"model/model.pt", "wb") as file:
        if self.regress:
            pickle.dump(self.regressor, file)
        else:
            pickle.dump(self.Q, file)
    with open("model/analysis_data.pt", "wb") as file:
        pickle.dump(self.analysis_data, file)

def updateQMatrix(self):
    tot_reward = 0
    for t in self.transitions:
        if t.action != None:
            tot_reward += t.reward

            if t.next_state is None:
                V = 0
            else:
                V = np.max(self.Q[tuple(t.next_state)])  # TODO: SARSA vs Q-Learning V
            action_index = ACTIONS.index(t.action)

            # get all symmetries
            origin_vec = np.concatenate((t.state, [action_index]))
            for rot in get_all_rotations(origin_vec):
                self.Q[tuple(rot)] += ALPHA * (
                    t.reward + GAMMA * V - self.Q[tuple(origin_vec)]
                )
    self.analysis_data["reward"].append(tot_reward)

def updateQ(self):
    batch = []
    occ_lengths = []  # for later slicing
    occasion = []  # storing transitions in occasions to reflect their context
    # measure reward
    tot_reward = 0
    n_features = 9
    states = np.ones((len(self.transitions), n_features), dtype=int)  # slightly too long still due to none states
    actions = np.ones((len(self.transitions)), dtype=int)
    rewards = np.ones((len(self.transitions)))
    cs = 0
    s = 0
    for i, t in enumerate(self.transitions):

        if t.state is not None:
            states[cs, :] = t.state
            actions[cs] = ACTIONS.index(t.action)
            rewards[cs] = t.reward
            tot_reward += t.reward
            cs += 1
            s += 1
            # TODO: prioritize interesting transitions

        else:
            tot_reward = 0
            if s > 0:
                self.analysis_data["reward"].append(tot_reward)
                occ_lengths.append(s)
                s = 0
    if s > 0:
        occ_lengths.append(s)
        self.analysis_data["reward"].append(tot_reward)

    del s, tot_reward

    ys = []
    xas = []
    states = states[:np.sum(occ_lengths)]  # cut none states
    actions = actions[:np.sum(occ_lengths)]
    rewards = rewards[:np.sum(occ_lengths)]

    predictions = np.reshape(self.regressor.predict_vec(states), (-1, 6))

    occ_pointer = 0
    for occ_l in occ_lengths:

        for i in range(occ_l):

            action = actions[occ_pointer + i]
            state = states[occ_pointer + i]
            rots = get_all_rotations(np.concatenate((state, [action])))
            for rot in rots:
                # calculate target response Y using n step TD!
                n = min(
                    occ_l - i - 1, N
                )  # calculate next N steps, otherwise just as far as possible
                r = [GAMMA ** k * rewards[occ_pointer + i + k] for k in range(n)]
                # TODO: Different Y models
                if states[occ_pointer + i + n] is not None:
                    Y = sum(r) + GAMMA ** n * np.max(predictions[occ_pointer + i + n][actions[occ_pointer + i + n]])
                    # Y = sum(r) + GAMMA ** n * np.max(self.regressor.predict(occ[i + n][0]))  # old
                else:
                    print("SOMETHING'S ROTTON IN updateQ()")
                # if t.next_state is not None:  # old
                #     Y = t.reward + GAMMA * np.max(self.regressor.predict(t.next_state))
                # else:
                #     Y = t.reward
                ys.append(Y)
                xas.append(rot)
                for a in range(len(ACTIONS)):
                    if a != rot[-1]:
                        ys.append(predictions[occ_pointer+i+n-1][actions[occ_pointer+i+n-1]])        # TODO: really this state not the next?
                        xas.append(np.concatenate((rot[:-1], [a])))
        occ_pointer += occ_l

    xas = np.array(xas)

    ys = np.array(ys)
    # print("Fitting xas", xas.shape, "ys", ys.shape)
    self.regressor.fit(xas, ys)


def reward_from_events(self, events: List[str]) -> int:
    """
    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    reward_sum = 0
    for event in events:
        if event in GAME_REWARDS:
            reward_sum += GAME_REWARDS[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
