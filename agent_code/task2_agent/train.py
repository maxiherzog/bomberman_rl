import pickle
import random
from collections import namedtuple, deque
from typing import List
import os
import scipy.sparse as sp

import events as e
from .callbacks import state_to_features, Q
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
ALPHA = 0.1
GAMMA = 0.9
XP_BUFFER_SIZE = 10

STORE_FREQ = XP_BUFFER_SIZE * 4

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
    self.rounds_played = 0

    if os.path.isfile("model.pt"):
        self.logger.info("Retraining from saved state.")
        with open("model.pt", "rb") as file:
            self.beta = pickle.load(file)
    else:
        self.logger.debug(f"Initializing Q")
        self.beta = np.zeros([8, len(ACTIONS)])

    # MEASUREING PARAMETERS
    if not os.path.exists("analysis"):
        os.makedirs("analysis")
    self.beta_dists = []
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

    :type self: object
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
    self.tot_rewards.append(tot_reward)
    self.beta_dists.append(np.sum(self.beta))
    self.crates_destroyed.append(self.crate_counter)
    self.crate_counter = 0

    self.rounds_played += 1
    if self.rounds_played % XP_BUFFER_SIZE == 0:
        updateQ(self)
        # clear transitions -> ready for next game
        self.transitions = deque(maxlen=None)

    if self.rounds_played % STORE_FREQ == 0:
        # Store the model
        with open(r"model.pt", "wb") as file:
            pickle.dump(self.beta, file)
        # Store analysis data
        with open("analysis/rewards.pt", "wb") as file:
            pickle.dump(self.tot_rewards, file)
        with open("analysis/beta-dists.pt", "wb") as file:
            pickle.dump(self.beta_dists, file)
        with open("analysis/crates.pt", "wb") as file:
            pickle.dump(self.crates_destroyed, file)
        with open("analysis/coins.pt", "wb") as file:
            pickle.dump(self.coins_collected, file)


def updateQ(self):
    batch = []
    for t in self.transitions:
        if t.state is not None:
            batch.append(t)  # TODO: prioritize interesting transitions


    for i in range(len(batch)):
        all_feat_action = get_all_rotations(np.concatenate([batch[i].state, [ACTIONS.index(batch[i].action)]]))
        for j in range(len(all_feat_action)):
            # calculate target response Y using TD # TODO: n-step TD
            if t.next_state is not None:
                Y = t.reward + GAMMA * np.max(Q(self, t.next_state))
            else:
                Y = t.reward
            # optimize Q towards Y
            state = np.array(all_feat_action[j][:-1])
            action = all_feat_action[j][-1]
            self.beta[:, action] += ALPHA/len(batch) * state * (Y - state.T @ self.beta[:, action])


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
