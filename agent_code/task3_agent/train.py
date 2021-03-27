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

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Events
EVADED_BOMB = "EVADED_BOMB"
NO_CRATE_OR_OPPONENT_DESTROYED = "NO_CRATE_OR_OPPONENT_DESTROYED"
NO_BOMB = "NO_BOMB"
BLOCKED_SELF_IN_UNSAFE_SPACE = "BLOCKED_SELF_IN_UNSAFE_SPACE"
DROPPED_BOMB_NEXT_TO_CRATE = "DROPPED_BOMB_NEXT_TO_CRATE"
NEW_PLACE = "NEW_PLACE"
NO_ACTIVE_BOMB = "NO_ACTIVE_BOMB"

# Hyper parameters -- DO modify
# TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
GAMMA = 0.90
ALPHA = 0.01
# N = 1  # for n-step TD Q learning
# XP_BUFFER_SIZE = 100  # higher batch size for forest
# N_ESTIMATORS = 100
# MAX_DEPTH = 40


EXPLOIT_SYMMETRY = True
GAME_REWARDS = {
    e.KILLED_OPPONENT: 2,
    e.COIN_COLLECTED: 1,
    e.INVALID_ACTION: -0.1,
    e.CRATE_DESTROYED: 0.4,
    e.KILLED_SELF: -0.5,
    e.BOMB_DROPPED: 0.02,
    DROPPED_BOMB_NEXT_TO_CRATE: 0.08,
    EVADED_BOMB: 0.1,
    NO_CRATE_OR_OPPONENT_DESTROYED: -0.7,
    NO_ACTIVE_BOMB: -0.07,
    BLOCKED_SELF_IN_UNSAFE_SPACE: -0.7,
}


STORE_FREQ = 100


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

    # ensure model subfolder
    if not os.path.exists("model"):
        os.makedirs("model")

    if os.path.isfile("model/model.pt"):
        self.logger.info("Retraining from saved state.")
        with open("model/model.pt", "rb") as file:
            self.Q = pickle.load(file)

        self.logger.info("Reloading analysis variables.")
        with open("model/analysis_data.pt", "rb") as file:
            self.analysis_data = pickle.load(file)
        print("WARNING: Cant use EPSILON_DECAY.. using EPSILON_MIN")
        self.epsilon = EPSILON_MIN
    else:
        self.logger.debug("Initializing Q")
        self.Q = np.zeros([2, 2, 2, 2, 2, 5, 5, 2, 2, 5, 5, 4, 2, len(ACTIONS)])
        print(np.prod(self.Q.shape))

        # if "AUTOTRAIN" in os.environ:
        #     if os.environ["AUTOTRAIN"] == "YES":
        #         ALPHA = os.environ["ALPHA"]
        #         GAMMA = os.environ["GAMMA"]

        # (save, POI_vector, [POI_type], [POI_dist], NEY_vector, [NEY_dist], [bomb_left])
        # dont run into walls
        self.Q[0, :, :, :, :, :, :, :, :, :, :, :, :, 0] += -2
        self.Q[:, 0, :, :, :, :, :, :, :, :, :, :, :, 1] += -2
        self.Q[:, :, 0, :, :, :, :, :, :, :, :, :, :, 2] += -2
        self.Q[:, :, :, 0, :, :, :, :, :, :, :, :, :, 3] += -2
        #
        # # drop Bomb when near crate
        self.Q[:, :, :, :, :, :, :, 0, 0, :, :, :, 1, 5] += 2

        # # dont drop Bomb when already having one bomb
        self.Q[:, :, :, :, :, :, :, :, :, :, :, :, 0, 5] += -2
        #
        # dont drop bomb when not near crate or enemy
        self.Q[:, :, :, :, :, :, :, 0, 1, :, :, 3, 1, 5] += -2
        # or near coin
        self.Q[:, :, :, :, :, :, :, 1, :, :, :, 3, 1, 5] += -2
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

        # walk away if not on save tile (only if safe)
        self.Q[1, :, :, :, 0, :, :, :, :, :, :, :, :, 0] += 3
        self.Q[:, 1, :, :, 0, :, :, :, :, :, :, :, :, 1] += 3
        self.Q[:, :, 1, :, 0, :, :, :, :, :, :, :, :, 2] += 3
        self.Q[:, :, :, 1, 0, :, :, :, :, :, :, :, :, 3] += 3

        # and dont fucking WAIT
        self.Q[:, :, :, :, 0, :, :, :, :, :, :, :, :, 4] += -2
        # but consider waiting if safe/dead
        self.Q[0, 0, 0, 0, 1, :, :, :, :, :, :, :, :, 4] += 2

        # xas = []  #  gamestate and action as argument
        # ys = []  # target response
        #
        # for i0 in range(Q.shape[0]):
        #     for i1 in range(Q.shape[1]):
        #         for i2 in range(Q.shape[2]):
        #             for i3 in range(Q.shape[3]):
        #                 for i4 in range(Q.shape[4]):
        #                     for i5 in range(Q.shape[5]):
        #                         for i6 in range(Q.shape[6]):
        #                             for i7 in range(Q.shape[7]):
        #                                 for i8 in range(Q.shape[8]):
        #                                     for a in range(Q.shape[8]):
        #                                         if (
        #                                             Q[
        #                                                 i0,
        #                                                 i1,
        #                                                 i2,
        #                                                 i3,
        #                                                 i4,
        #                                                 i5,
        #                                                 i6,
        #                                                 i7,
        #                                                 i8,
        #                                                 a,
        #                                             ]
        #                                             != 0
        #                                         ):
        #                                             xas.append(
        #                                                 [
        #                                                     i0,
        #                                                     i1,
        #                                                     i2,
        #                                                     i3,
        #                                                     i4,
        #                                                     i5,
        #                                                     i6,
        #                                                     i7,
        #                                                     i8,
        #                                                     a,
        #                                                 ]
        #                                             )
        #                                             ys.append(
        #                                                 Q[
        #                                                     i0,
        #                                                     i1,
        #                                                     i2,
        #                                                     i3,
        #                                                     i4,
        #                                                     i5,
        #                                                     i6,
        #                                                     i7,
        #                                                     i8,
        #                                                     a,
        #                                                 ]
        #                                             )
        # self.regressor = Network(8)
        # self.regressor = Forest(
        #     9, n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=0
        # )
        # # print(
        # #     "Fitting forest.. dofs:",
        # #     2 ** MAX_DEPTH * N_ESTIMATORS,
        # #     "vs.",
        # #     np.prod(Q.shape),
        # # )
        # xas = np.array(xas)
        # ys = np.array(ys)
        # # print("Fitting xas", xas.shape, "ys", ys.shape)
        # self.regressor.fit(xas, ys)
        # print(self.forest.predict(np.reshape([0, 0, 0, 0, 0, 0, 0, 0, 4], (1, -1))))
        # init measured variables
        self.analysis_data = {
            # "Q_sum": [],
            # "Q_sum_move": [],
            # "Q_sum_bomb": [],
            # "Q_sum_wait": [],
            # "Q_situation": [self.Q[0, 1, 1, 0, 1, 2, 1, 2, :]],
            "reward": [],
            "win": [],
            "kills": [],
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
            # "XP_BUFFER_SIZE": XP_BUFFER_SIZE,
            # "N": N,
            # "EXPLOIT_SYMMETRY": EXPLOIT_SYMMETRY,
            # "REGRESSION TREE": {"N_ESTIMATORS": N_ESTIMATORS, "MAX_DEPTH": MAX_DEPTH,},
            "GAME_REWARDS": GAME_REWARDS,
        }
        with open("model/hyperparams.json", "w") as file:
            json.dump(hyperparams, file, ensure_ascii=False, indent=4)
        store(self)

    # init counters
    # hands on variables
    self.kill_counter = 0
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

    if (
        e.BOMB_EXPLODED in events
        and not e.CRATE_DESTROYED in events
        and not e.KILLED_OPPONENT in events
    ):
        events.append(NO_CRATE_OR_OPPONENT_DESTROYED)
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
    if all(new_feat[:5] == 0):
        events.append(BLOCKED_SELF_IN_UNSAFE_SPACE)
    if old_feat is not None:
        if not (old_feat[12] == 1 or new_feat[12] == 1):
            events.append(NO_ACTIVE_BOMB)
    else:
        events.append(NO_ACTIVE_BOMB)

    # if e.MOVED_UP or e.MOVED_DOWN or e.MOVED_LEFT or e.MOVED_RIGHT:
    #     if new_game_state["self"][0] not in self.visited:
    #         self.visited.append(new_game_state["self"][3])
    #         events.append(NEW_PLACE)

    # DROPPED_BOMB_NEXT_TO_CRATE
    if e.BOMB_DROPPED in events:
        if old_feat[7] == 0 and old_feat[8] == 0:
            events.append(DROPPED_BOMB_NEXT_TO_CRATE)

    if e.KILLED_OPPONENT in events:
        self.kill_counter += 1

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
    # ADJUST EPSILON
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

    self.analysis_data["kills"].append(self.kill_counter)
    self.analysis_data["win"].append(self.kill_counter == 3)
    self.analysis_data["crates"].append(self.crate_counter)
    self.analysis_data["coins"].append(self.coin_counter)
    self.analysis_data["length"].append(last_game_state["step"])
    self.analysis_data["bombs"].append(self.bombs_counter)
    self.analysis_data["useless_bombs"].append(self.useless_bombs_counter)

    # RESET Counters
    self.crate_counter = 0
    self.kill_counter = 0
    self.coin_counter = 0
    self.bombs_counter = 0
    self.useless_bombs_counter = 0

    self.rounds_played += 1
    # if self.rounds_played % XP_BUFFER_SIZE == 0:
    #     updateQ(self)
    #     # clear transitions -> ready for next game
    #     self.transitions = deque(maxlen=None)

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

    if last_game_state["round"] % STORE_FREQ == 0:
        store(self)
    self.transitions = deque(maxlen=None)


def store(self):
    """
    Stores all the files.
    """
    # Store the model
    self.logger.debug("Storing model.")
    with open(r"model/model.pt", "wb") as file:
        pickle.dump(self.Q, file)
    with open("model/analysis_data.pt", "wb") as file:
        pickle.dump(self.analysis_data, file)


# def updateQ(self):
#     batch = []
#     occasion = []  # storing transitions in occasions to reflect their context
#     # measure reward
#     tot_reward = 0
#     for t in self.transitions:
#         if t.state is not None:
#             tot_reward += t.reward
#             occasion.append(t)  # TODO: prioritize interesting transitions
#         else:
#             self.analysis_data["reward"].append(tot_reward)
#             tot_reward = 0
#             batch.append(occasion)
#             occasion = []
#
#     ys = []
#     xas = []
#     for occ in batch:
#         np.random.shuffle(occ)
#         for i, t in enumerate(occ):
#             action = [ACTIONS.index(t.action)]
#             rots = get_all_rotations(np.concatenate((t.state, action)))
#             for rot in rots:
#                 # calculate target response Y using n step TD!
#                 # n = min(
#                 #     len(occ) - i, N
#                 # )  # calculate next N steps, otherwise just as far as possible
#                 # r = [GAMMA ** k * occ[i + k].reward for k in range(n)]
#                 # TODO: Different Y models
#                 if t.next_state is not None:
#                     # Y = sum(r) + GAMMA ** n * np.max(Q(self, t.next_state))
#                     Y = t.reward + GAMMA * np.max(self.regressor.predict(t.next_state))
#                 else:
#                     Y = t.reward
#                 ys.append(Y)
#                 xas.append(rot)
#                 for a in range(len(ACTIONS)):
#                     if a != rot[-1]:
#                         ys.append(self.regressor.predict(t.state)[a])
#                         xas.append(np.concatenate((rot[:-1], [a])))
#
#     xas = np.array(xas)
#
#     ys = np.array(ys)
#     # print("Fitting xas", xas.shape, "ys", ys.shape)
#     self.regressor.fit(xas, ys)


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
