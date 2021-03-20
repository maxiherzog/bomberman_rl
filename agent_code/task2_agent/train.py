import pickle
import random
from collections import namedtuple, deque
from typing import List
import os
import scipy.sparse as sp

import json

import events as e
from .callbacks import state_to_features
from .callbacks import ACTIONS
from .callbacks import get_all_rotations
from .callbacks import EPSILON
import numpy as np
from .sparseTensor import SparseTensor


import time


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Events
EVADED_BOMB = "EVADED_BOMB"
NO_CRATE_DESTROYED = "NO_CRATE_DESTROYED"
NO_BOMB = "NO_BOMB"


# Hyper parameters -- DO modify
# TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
ALPHA = 0.1
GAMMA = 0.8
GAME_REWARDS = {
    e.COIN_COLLECTED: 2,
    # e.KILLED_OPPONENT: 5,
    e.INVALID_ACTION: -1,
    e.CRATE_DESTROYED: 1,
    e.KILLED_SELF: -2,
    # e.BOMB_DROPPED: 0.5,
    # EVADED_BOMB: 1,
    NO_BOMB: -0.05,
    NO_CRATE_DESTROYED: -0.5,
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
    if not os.path.exists("analysis"):
        os.makedirs("analysis")

    if os.path.isfile("model.pt"):
        self.logger.info("Retraining from saved state.")
        with open("model.pt", "rb") as file:
            self.Q = pickle.load(file)

        self.logger.info("Reloading analysis variables.")
        with open("analysis/Q-dists.pt", "rb") as file:
            self.Q_dists = pickle.load(file)
        with open("analysis/rewards.pt", "rb") as file:
            self.tot_rewards = pickle.load(file)
        with open("analysis/coins.pt", "rb") as file:
            self.coins_collected = pickle.load(file)
        with open("analysis/crates.pt", "rb") as file:
            self.crates_destroyed = pickle.load(file)
    else:
        self.logger.debug(f"Initializing Q")
        self.Q = np.zeros([2, 2, 2, 2, 3, 3, 3, 5, len(ACTIONS)])

        self.Q[0, :, :, :, :, :, :, :, 0] = -1
        self.Q[:, 0, :, :, :, :, :, :, 1] = -1
        self.Q[:, :, 0, :, :, :, :, :, 2] = -1
        self.Q[:, :, :, 0, :, :, :, :, 3] = -1

        # init measured variables
        self.Q_dists = []
        self.tot_rewards = []
        self.coins_collected = []
        self.crates_destroyed = []

        # dump hyper parameters as json
        hyperparams = {
            "ALPHA": ALPHA,
            "GAMMA": GAMMA,
            "EPSILON": EPSILON,
            "GAME_REWARDS": GAME_REWARDS,
        }
        with open("analysis/hyperparams.json", "w") as file:
            json.dump(hyperparams, file, ensure_ascii=False, indent=4)

    # init counters
    # hands on variables
    self.crate_counter = 0
    self.coin_counter = 0


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

    if not e.BOMB_DROPPED in events:
        events.append(NO_BOMB)

    if e.CRATE_DESTROYED in events:
        self.crate_counter += 1

    if e.COIN_COLLECTED in events:
        self.coin_counter += 1

    # state_to_features is defined in callbacks.py
    self.transitions.append(
        Transition(
            state_to_features(old_game_state),
            self_action,
            state_to_features(new_game_state),
            reward_from_events(self, events),
        )
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

    tot_reward = 0

    for trans in self.transitions:
        if trans.action != None:
            tot_reward += trans.reward

            if trans.next_state is None:
                V = 0
            else:
                V = np.max(
                    self.Q[tuple(trans.next_state)]
                )  # TODO: SARSA vs Q-Learning V
            action_index = ACTIONS.index(trans.action)

            # get all symmetries
            origin_vec = np.concatenate((trans.state, [action_index]))
            # encountered_symmetry = False
            for rot in get_all_rotations(origin_vec):
                self.Q[tuple(rot)] += ALPHA * (
                    trans.reward + GAMMA * V - self.Q[tuple(origin_vec)]
                )

    self.tot_rewards.append(tot_reward)
    self.Q_dists.append(np.sum(self.Q))

    self.crates_destroyed.append(self.crate_counter)
    self.crate_counter = 0

    self.coins_collected.append(self.coin_counter)
    self.coin_counter = 0

    if last_game_state["round"] % 500 == 0:
        store(self)
    # clear transitions -> ready for next game
    self.transitions = deque(maxlen=None)


def store(self):
    """
    Stores all the files.
    """
    # Store the model
    self.logger.debug("Storing model.")
    with open(r"model.pt", "wb") as file:
        pickle.dump(self.Q, file)
    with open("analysis/rewards.pt", "wb") as file:
        pickle.dump(self.tot_rewards, file)
    with open("analysis/Q-dists.pt", "wb") as file:
        pickle.dump(self.Q_dists, file)
    with open("analysis/crates.pt", "wb") as file:
        pickle.dump(self.crates_destroyed, file)
    with open("analysis/coins.pt", "wb") as file:
        pickle.dump(self.coins_collected, file)


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
