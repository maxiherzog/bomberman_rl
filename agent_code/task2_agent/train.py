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
from .callbacks import Q
import numpy as np
from .sparseTensor import SparseTensor


import time


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Events
EVADED_BOMB = "EVADED_BOMB"
NO_CRATE_DESTROYED = "NO_CRATE_DESTROYED"
NO_BOMB = "NO_BOMB"
NEW_PLACE = "NEW_PLACE"


# Hyper parameters -- DO modify
# TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
ALPHA = 0.2
GAMMA = 0.9
N = 4           # for n-step TD Q learning
XP_BUFFER_SIZE = 2

GAME_REWARDS = {
    e.COIN_COLLECTED: 1,
    # e.KILLED_OPPONENT: 5,
    e.INVALID_ACTION: -0.2,
    e.CRATE_DESTROYED: 0.5,
    e.KILLED_SELF: -1,
    e.WAITED: -0.2,
    NEW_PLACE: 0.1,
    # e.BOMB_DROPPED: 0.5,
    # EVADED_BOMB: 1,
    #NO_BOMB: -0.05,
    #NO_CRATE_DESTROYED: -0.5,
}


STORE_FREQ = 500

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

    # ensure analysis subfolder
    if not os.path.exists("analysis"):
        os.makedirs("analysis")

    if os.path.isfile("model.pt"):
        self.logger.info("Retraining from saved state.")
        with open("model.pt", "rb") as file:
            self.beta = pickle.load(file)
        self.logger.info("Reloading analysis variables.")
        with open("analysis/beta-dists.pt", "rb") as file:
            self.beta_dists = pickle.load(file)
        with open("analysis/rewards.pt", "rb") as file:
            self.tot_rewards = pickle.load(file)
        with open("analysis/coins.pt", "rb") as file:
            self.coins_collected = pickle.load(file)
        with open("analysis/crates.pt", "rb") as file:
            self.crates_destroyed = pickle.load(file)
    else:
        self.logger.debug(f"Initializing Q")
        self.beta = np.zeros([8, len(ACTIONS)])


        # init measured variables
        self.beta_dists = []
        self.tot_rewards = []
        self.coins_collected = []
        self.crates_destroyed = []

        # dump hyper parameters as json
        hyperparams = {
            "ALPHA": ALPHA,
            "GAMMA": GAMMA,
            "EPSILON": EPSILON,
            "XP_BUFFER_SIZE": XP_BUFFER_SIZE,
            "N": N,
            "GAME_REWARDS": GAME_REWARDS,
        }
        with open("analysis/hyperparams.json", "w") as file:
            json.dump(hyperparams, file, ensure_ascii=False, indent=4)

    # init counters
    # hands on variables
    self.crate_counter = 0
    self.coin_counter = 0
    self.visited = []

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

    if e.MOVED_UP or e.MOVED_DOWN or e.MOVED_LEFT or e.MOVED_RIGHT:
        if new_game_state["self"][0] not in self.visited:
            self.visited.append(new_game_state["self"][0])
            events.append(NEW_PLACE)

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

    self.coins_collected.append(self.coin_counter)
    self.coin_counter = 0

    self.visited = []

    self.rounds_played += 1
    if self.rounds_played % XP_BUFFER_SIZE == 0:
        updateQ(self)
        # clear transitions -> ready for next game
        self.transitions = deque(maxlen=None)

    if last_game_state["round"] % STORE_FREQ == 0:
        store(self)

def store(self):
    """
    Stores all the files.
    """
    # Store the model
    self.logger.debug("Storing model.")
    with open(r"model.pt", "wb") as file:
        pickle.dump(self.beta, file)
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
    occasion = []               # storing transitions in occasions to reflect their context
    for t in self.transitions:
        if t.state is not None:
            occasion.append(t)  # TODO: prioritize interesting transitions
        else:
            batch.append(occasion)
            occasion = []

    for occ in batch:
        for i,t in enumerate(occ):
            all_feat_action = get_all_rotations(np.concatenate([occ[i].state, [ACTIONS.index(occ[i].action)]]))
            for j in range(len(all_feat_action)):
                # calculate target response Y using n step TD!
                n = min(len(occ)-i, N)      # calculate next N steps, otherwise just as far as possible
                r = [GAMMA**k * occ[i+k].reward for k in range(n)]
                if t.next_state is not None:
                    Y = sum(r) + GAMMA ** n * np.max(Q(self, t.next_state))
                else:
                    Y = t.reward
                # optimize Q towards Y
                state = np.array(all_feat_action[j][:-1])
                action = all_feat_action[j][-1]
                self.beta[:, action] += ALPHA/len(self.transitions) * state * (Y - state.T @ self.beta[:, action])       # TODO: think about batch size division


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
