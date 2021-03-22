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
from .callbacks import Q
import numpy as np


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Events
EVADED_BOMB = "EVADED_BOMB"
NO_CRATE_DESTROYED = "NO_CRATE_DESTROYED"
NO_BOMB = "NO_BOMB"
BLOCKED_SELF_IN_UNSAFE_SPACE = "BLOCKED_SELF_IN_UNSAFE_SPACE"
DROPPED_BOMB_NEXT_TO_CRATE = "DROPPED_BOMB_NEXT_TO_CRATE"
NEW_PLACE = "NEW_PLACE"


# Hyper parameters -- DO modify
# TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
ALPHA = 0.02
GAMMA = 0.9
N = 2  # for n-step TD Q learning
XP_BUFFER_SIZE = 10     # higher batch size for forest

EXPLOIT_SYMMETRY = True
GAME_REWARDS = {
    # HANS
    e.COIN_COLLECTED: 1,
    e.INVALID_ACTION: -0.1,
    e.CRATE_DESTROYED: 0.5,
    e.KILLED_SELF: -0.5,
    e.BOMB_DROPPED: 0.05,
    EVADED_BOMB: 0.1,
    NO_CRATE_DESTROYED: -0.2,
    DROPPED_BOMB_NEXT_TO_CRATE: 0.1,
    NO_BOMB: -0.05,
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


STORE_FREQ = 50


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

    else:
        self.logger.debug("Initializing Q")
        # self.Q = np.zeros([2, 2, 2, 2, 5, 5, 3, 5, len(ACTIONS)])
        # 
        # # dont run into walls
        # self.Q[0, :, :, :, :, :, :, :, 0] += -2
        # self.Q[:, 0, :, :, :, :, :, :, 1] += -2
        # self.Q[:, :, 0, :, :, :, :, :, 2] += -2
        # self.Q[:, :, :, 0, :, :, :, :, 3] += -2
        # 
        # # drop Bomb when near crate
        # self.Q[:, :, :, :, :, :, 1, 1, 5] += 1
        # # dont drop Bomb when already having one bomb
        # self.Q[:, :, :, :, :, :, 0, :, 5] += -2
        # 
        # # dont drop bomb when not near crate
        # self.Q[:, :, :, :, :, :, 1, 2:, 5] += -2
        # 
        # # walk towards crates
        # self.Q[1, :, :, :, :, :2, 1, 2:, 0] += 1
        # self.Q[:, 1, :, :, -2:, :, 1, 2:, 1] += 1
        # self.Q[:, :, 1, :, :, -2:, 1, 2:, 2] += 1
        # self.Q[:, :, :, 1, :2, :, 1, 2:, 3] += 1
        # 
        # # walk towards coins
        # self.Q[1, :, :, :, :, :2, 2, :, 0] += 1
        # self.Q[:, 1, :, :, -2:, :, 2, :, 1] += 1
        # self.Q[:, :, 1, :, :, -2:, 2, :, 2] += 1
        # self.Q[:, :, :, 1, :2, :, 2, :, 3] += 1
        # 
        # # walk away from bomb (only if safe) if ON BOMB
        # self.Q[1, :, :, :, :, :, 0, :, 0] += 1
        # self.Q[:, 1, :, :, :, :, 0, :, 1] += 1
        # self.Q[:, :, 1, :, :, :, 0, :, 2] += 1
        # self.Q[:, :, :, 1, :, :, 0, :, 3] += 1
        # 
        # # and in straight lines
        # # self.Q[:, :, 1, :, 1, 0, 0, 1:, 2] += 1
        # # self.Q[:, :, :, 1, 2, 1, 0, 1:, 3] += 1
        # # self.Q[1, :, :, :, 1, 2, 0, 1:, 0] += 1
        # # self.Q[:, 1, :, :, 0, 1, 0, 1:, 1] += 1
        # 
        # # and dont fucking WAIT
        # self.Q[:, :, :, :, :, :, 0, :, 4] += -1
        # # but consider waiting if safe/dead
        # self.Q[0, 0, 0, 0, :, :, 0, :, 4] += 1

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
            "XP_BUFFER_SIZE": XP_BUFFER_SIZE,
            "N": N,
            "GAME_REWARDS": GAME_REWARDS,
            "EXPLOIT_SYMMETRY": EXPLOIT_SYMMETRY,
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
    if e.MOVED_UP or e.MOVED_DOWN or e.MOVED_LEFT or e.MOVED_RIGHT:
        if new_game_state["self"][0] not in self.visited:
            self.visited.append(new_game_state["self"][3])
            events.append(NEW_PLACE)

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
    
    # measure
    tot_reward = 0
    for trans in self.transitions:
        if trans.action is not None:
            tot_reward += trans.reward

    self.analysis_data["reward"].append(tot_reward)
    self.analysis_data["Q_sum"].append(np.sum(self.Q))
    
    # RESET Counters
    self.crate_counter = 0
    self.coin_counter = 0
    self.useless_bombs_counter = 0
    
    self.analysis_data["crates"].append(self.crate_counter)
    self.analysis_data["coins"].append(self.coin_counter)
    self.analysis_data["length"].append(last_game_state["step"])
    self.analysis_data["useless_bombs"].append(self.useless_bombs_counter)

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
    with open(r"model/model.pt", "wb") as file:
        pickle.dump(self.Q, file)
    with open("model/analysis_data.pt", "wb") as file:
        pickle.dump(self.analysis_data, file)


def updateQ(self):
    batch = []
    occasion = []  # storing transitions in occasions to reflect their context
    for t in self.transitions:
        if t.state is not None:
            occasion.append(t)  # TODO: prioritize interesting transitions
        else:
            batch.append(occasion)
            occasion = []
    ys = []
    xas = []
    for occ in batch:
        for i, t in enumerate(occ):
            all_feat_action = get_all_rotations(
                np.concatenate([occ[i].state, [ACTIONS.index(occ[i].action)]])
            )
            for j in range(len(all_feat_action)):
                # calculate target response Y using n step TD!
                n = min(
                    len(occ) - i, N
                )  # calculate next N steps, otherwise just as far as possible
                r = [GAMMA ** k * occ[i + k].reward for k in range(n)]
                # TODO: Different Y models
                if t.next_state is not None:
                    Y = sum(r) + GAMMA ** n * np.max(Q(self, t.next_state))
                else:
                    Y = t.reward
                ys.append(Y)
                xas.append(all_feat_action[j])
    xas = np.array(xas)
    ys = np.array(ys)
    self.forest.fit(xas, ys)


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
