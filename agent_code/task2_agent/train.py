import pickle
from collections import namedtuple, deque
from typing import List
import os

import json

import events as e
from .callbacks import state_to_features
from .callbacks import ACTIONS
from .callbacks import get_all_rotations
from .callbacks import EPSILON
import numpy as np


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Events
EVADED_BOMB = "EVADED_BOMB"
NO_CRATE_DESTROYED = "NO_CRATE_DESTROYED"
NO_BOMB = "NO_BOMB"
BLOCKED_SELF_IN_UNSAFE_SPACE = "BLOCKED_SELF_IN_UNSAFE_SPACE"
DROPPED_BOMB_NEXT_TO_CRATE = "DROPPED_BOMB_NEXT_TO_CRATE"
# Hyper parameters -- DO modify
# TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
ALPHA = 0.02
GAMMA = 0.9
GAME_REWARDS = {
    # HANS
    # e.COIN_COLLECTED: 1,
    # e.INVALID_ACTION: -0.1,
    # e.CRATE_DESTROYED: 0.5,
    # e.KILLED_SELF: -0.5,
    # e.BOMB_DROPPED: 0.05,
    # EVADED_BOMB: 0.1,
    # NO_CRATE_DESTROYED: -0.1,
    # NO_BOMB: -0.05,
    # BLOCKED_SELF_IN_UNSAFE_SPACE: -0.3,
    # MAXI
    e.COIN_COLLECTED: 1,
    e.INVALID_ACTION: -1,
    BLOCKED_SELF_IN_UNSAFE_SPACE: -10,
    e.CRATE_DESTROYED: 0.1,
    NO_BOMB: -0.05,
    e.NO_CRATE_DESTROYED: -3
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


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=None)  #

    # ensure analysis subfolder
    if not os.path.exists("model"):
        os.makedirs("model")

    if os.path.isfile("model/model.pt"):
        self.logger.info("Retraining from saved state.")
        with open("model/model.pt", "rb") as file:
            self.Q = pickle.load(file)

        self.logger.info("Reloading analysis variables.")
        with open("model/analysis_data.pt", "rb") as file:
            self.analysis_data = pickle.load(file)

    else:
        self.logger.debug("Initializing Q")
        self.Q = np.zeros([2, 2, 2, 2, 5, 5, 3, 5, len(ACTIONS)])

        # dont run into walls
        self.Q[0, :, :, :, :, :, :, :, 0] += -2
        self.Q[:, 0, :, :, :, :, :, :, 1] += -2
        self.Q[:, :, 0, :, :, :, :, :, 2] += -2
        self.Q[:, :, :, 0, :, :, :, :, 3] += -2

        # drop Bomb when near crate
        self.Q[:, :, :, :, :, :, 1, 1, 5] += 1
        # dont drop Bomb when already having one bomb
        self.Q[:, :, :, :, :, :, 0, :, 5] += -2

        # dont drop bomb when not near crate
        self.Q[:, :, :, :, :, :, 1, 2:, 5] += -2

        # walk towards crates
        self.Q[1, :, :, :, :, :2, 1, 2:, 0] += 1
        self.Q[:, 1, :, :, -2:, :, 1, 2:, 1] += 1
        self.Q[:, :, 1, :, :, -2:, 1, 2:, 2] += 1
        self.Q[:, :, :, 1, :2, :, 1, 2:, 3] += 1

        # walk towards coins
        self.Q[1, :, :, :, :, :2, 2, :, 0] += 1
        self.Q[:, 1, :, :, -2:, :, 2, :, 1] += 1
        self.Q[:, :, 1, :, :, -2:, 2, :, 2] += 1
        self.Q[:, :, :, 1, :2, :, 2, :, 3] += 1

        # walk away from bomb (only if safe) if ON BOMB
        self.Q[1, :, :, :, :, :, 0, :, 0] += 1
        self.Q[:, 1, :, :, :, :, 0, :, 1] += 1
        self.Q[:, :, 1, :, :, :, 0, :, 2] += 1
        self.Q[:, :, :, 1, :, :, 0, :, 3] += 1

        # and in straight lines
        # self.Q[:, :, 1, :, 1, 0, 0, 1:, 2] += 1
        # self.Q[:, :, :, 1, 2, 1, 0, 1:, 3] += 1
        # self.Q[1, :, :, :, 1, 2, 0, 1:, 0] += 1
        # self.Q[:, 1, :, :, 0, 1, 0, 1:, 1] += 1

        # and dont fucking WAIT
        self.Q[:, :, :, :, :, :, 0, :, 4] += -1
        # but consider waiting if safe/dead
        self.Q[0, 0, 0, 0, :, :, 0, :, 4] += 1

        # init measured variables
        self.analysis_data = {
            "Q_sum": [],
            # "Q_sum_move": [],
            # "Q_sum_bomb": [],
            # "Q_sum_wait": [],
            # "Q_situation": [self.Q[0, 1, 1, 0, 1, 2, 1, 2, :]],
            "reward": [],
            "coins": [],
            "crates": [],
            "length": [],
            "useless_bombs": [],
        }

        # dump hyper parameters as json
        hyperparams = {
            "ALPHA": ALPHA,
            "GAMMA": GAMMA,
            "EPSILON": EPSILON,
            "GAME_REWARDS": GAME_REWARDS,
        }
        with open("model/hyperparams.json", "w") as file:
            json.dump(hyperparams, file, ensure_ascii=False, indent=4)

    # init counters
    # hands on variables
    self.crate_counter = 0
    self.coin_counter = 0
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

    # DROPPED_BOMB_NEXT_TO_CRATE
    if e.BOMB_DROPPED in events:
        if old_feat[6] == 1 and old_feat[7] == 1:
            events.append(DROPPED_BOMB_NEXT_TO_CRATE)

    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(old_feat, self_action, new_feat, reward_from_events(self, events),)
    )


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
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

    # UPDATE Q
    updateQ(self)

    # measure total reward
    tot_reward = 0
    for trans in self.transitions:
        if trans.action != None:
            tot_reward += trans.reward

    self.analysis_data["reward"].append(tot_reward)
    self.analysis_data["Q_sum"].append(np.sum(self.Q))
    # self.analysis_data["Q_sum_move"].append(np.sum(self.Q[:, :, :, :, :, :, :, :, 0]))
    # self.analysis_data["Q_sum_bomb"].append(np.sum(self.Q[:, :, :, :, :, :, :, :, 5]))
    # self.analysis_data["Q_sum_wait"].append(np.sum(self.Q[:, :, :, :, :, :, :, :, 4]))
    # self.analysis_data["Q_situation"].append(self.Q[0, 1, 1, 0, 1, 2, 1, 2, :])
    # print(self.Q[0, 1, 1, 0, 1, 2, 1, 2, :])
    self.analysis_data["crates"].append(self.crate_counter)
    self.analysis_data["coins"].append(self.coin_counter)
    self.analysis_data["length"].append(last_game_state["step"])
    self.analysis_data["useless_bombs"].append(self.useless_bombs_counter)

    if last_game_state["round"] % 100 == 0:
        store(self)
    # clear transitions -> ready for next game
    self.transitions = deque(maxlen=None)
    # RESET Counters
    self.crate_counter = 0
    self.coin_counter = 0
    self.useless_bombs_counter = 0


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


def updateQ(self):
    """
    Updates the Q function.
    """
    for trans in self.transitions:
        if trans.action != None:

            if trans.next_state is None:
                V = 0
            else:
                V = np.max(
                    self.Q[tuple(trans.next_state)]
                )  # TODO: SARSA vs Q-Learning V
            action_index = ACTIONS.index(trans.action)

            # get all symmetries
            origin_vec = np.concatenate((trans.state, [action_index]))
            for rot in get_all_rotations(origin_vec):
                self.Q[tuple(rot)] += ALPHA * (
                    trans.reward + GAMMA * V - self.Q[tuple(origin_vec)]
                )


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
