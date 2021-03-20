import pickle
import random
from collections import namedtuple, deque
from typing import List
import os
import scipy.sparse as sp

import events as e
from .callbacks import state_to_features
from .callbacks import ACTIONS
from .callbacks import get_all_rotations
import numpy as np
from .sparseTensor import SparseTensor

import time


# This is only an example!
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Hyper parameters -- DO modify
# TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
EVADED_BOMB = "EVADED_BOMB"
NO_CRATE_DESTROYED = "NO_CRATE_DESTROYED"
NO_BOMB = "NO_BOMB"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=None)  #

    if os.path.isfile("model.pt"):
        self.logger.info("Retraining from saved state.")
        with open("model.pt", "rb") as file:
            self.Q = pickle.load(file)
    else:
        self.logger.debug(f"Initializing Q")
        self.Q = np.zeros([2, 2, 2, 2, 3, 3, 3, 5, len(ACTIONS)])

        self.Q[0, :, :, :, :, :, :, :, 0] = -1
        self.Q[:, 0, :, :, :, :, :, :, 1] = -1
        self.Q[:, :, 0, :, :, :, :, :, 2] = -1
        self.Q[:, :, :, 0, :, :, :, :, 3] = -1

    # MEASUREING PARAMETERS

    if not os.path.exists("analysis"):
        os.makedirs("analysis")
    self.Q_dists = []
    self.tot_rewards = []

    # hands on variables
    self.crate_counter = 0
    self.crates_destroyed = []
    self.coin_counter = 0
    self.coins_collected = []


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
        self.crate_counter += 1

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
            alpha = 0.1
            gamma = 0.94
            action_index = ACTIONS.index(trans.action)

            # get all symmetries
            origin_vec = np.concatenate((trans.state, [action_index]))
            # encountered_symmetry = False
            for rot in get_all_rotations(origin_vec):
                self.Q[tuple(rot)] += alpha * (
                    trans.reward + gamma * V - self.Q[tuple(origin_vec)]
                )

    # Store the model
    with open(r"model.pt", "wb") as file:
        pickle.dump(self.Q, file)

    self.tot_rewards.append(tot_reward)
    with open("analysis/rewards.pt", "wb") as file:
        pickle.dump(self.tot_rewards, file)
    self.Q_dists.append(np.sum(self.Q))
    with open("analysis/Q-dists.pt", "wb") as file:
        pickle.dump(self.Q_dists, file)

    self.crates_destroyed.append(self.crate_counter)
    self.crate_counter = 0
    with open("analysis/crates.pt", "wb") as file:
        pickle.dump(self.crates_destroyed, file)
    self.coins_collected.append(self.coin_counter)
    self.coin_counter = 0
    with open("analysis/coins.pt", "wb") as file:
        pickle.dump(self.coins_collected, file)

    # clear transitions -> ready for next game
    self.transitions = deque(maxlen=None)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 3,
        # e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -1,
        e.CRATE_DESTROYED: 2,
        e.KILLED_SELF: -1,
        e.BOMB_DROPPED: 0.3,
        EVADED_BOMB: 1,
        NO_BOMB: -0.05,
        NO_CRATE_DESTROYED: -1.4,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
