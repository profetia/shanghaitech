import copy
from typing import Any, List, Tuple
import numpy as np
from collections import OrderedDict

from .game import (
    HoldemGame,
    HoldemGameState,
    HoldemPlayerAction,
    action_lookup,
)
from .card import card_map
from ..utils.random import create_random_generator
from ..models.agent import Agent


def convert_action(
    action_id: int, legal_actions: List[HoldemPlayerAction]
) -> HoldemPlayerAction:
    if action_lookup[action_id] not in legal_actions:
        if HoldemPlayerAction.CHECK in legal_actions:
            return HoldemPlayerAction.CHECK
        else:
            return HoldemPlayerAction.FOLD
    return action_lookup[action_id]


class Env:
    def __init__(
        self,
        agents: List[Agent],
        num_players: int = 2,
        seed: int = 3407,
    ) -> None:
        random, _ = create_random_generator(seed)

        self.agents = agents
        self.game = HoldemGame(
            num_players=num_players,
            random=random,
        )

        self.timestep = 0
        self.action_recorder: List[Tuple[int, HoldemPlayerAction]] = []

    @property
    def num_players(self) -> int:
        return self.game.num_players

    def reset(self) -> Tuple[HoldemGameState, int]:
        state, player_id = self.game.reset()
        self.action_recorder = []
        return self.__extract_state(state), player_id

    def step(self, action: int) -> Tuple[HoldemGameState, int]:
        converted_action: HoldemPlayerAction = convert_action(
            action, self.game.legal_actions()
        )
        self.timestep += 1
        self.action_recorder.append((self.game.turn_id, converted_action))
        next_state, player_id = self.game.step(converted_action)
        return self.__extract_state(next_state), player_id

    def run(
        self, is_training=False
    ) -> Tuple[int, List[List[HoldemGameState | int]], List[float]]:
        trajectories: List[List[HoldemGameState | int]] = [
            [] for _ in range(self.game.num_players)
        ]
        state: HoldemGameState
        starter_id: int
        state, starter_id = self.reset()

        player_id: int = starter_id
        trajectories[player_id].append(state)
        while not self.is_over():
            action: int
            if not is_training:
                action, _ = self.agents[player_id].eval_step(state)
            else:
                action = self.agents[player_id].step(state)

            next_state: HoldemGameState
            next_player_id: int
            next_state, next_player_id = self.step(action)

            trajectories[player_id].append(action)

            state = next_state
            player_id = next_player_id

            if not self.game.is_over():
                trajectories[player_id].append(state)

        for player_id in range(self.game.num_players):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        payoffs = self.game.payoffs()
        return starter_id, trajectories, payoffs

    def is_over(self) -> bool:
        return self.game.is_over()

    def get_state(self, player_id) -> HoldemGameState:
        return self.__extract_state(self.game.get_state(player_id))

    def __extract_state(self, state: HoldemGameState) -> HoldemGameState:
        result: HoldemGameState = {}

        obs = np.zeros(72)
        obs[
            [card_map[card] for card in state["public_cards"] + state["hand"]]
        ] = 1
        for i, num in enumerate(state["raise_nums"]):
            obs[52 + i * 5 + num] = 1
        result["obs"] = obs
        result["legal_actions"] = OrderedDict(
            {
                action_lookup.index(action): None
                for action in state["legal_actions"]
            }
        )

        result["raw_obs"] = state
        result["raw_legal_actions"] = [
            action for action in state["legal_actions"]
        ]
        result["action_record"] = self.action_recorder

        return result


def evaluate_perf(env: Env, iters: int) -> List[float]:
    payoffs: List[float] = [0 for _ in range(env.num_players)]
    for _ in range(iters):
        _, _, earns = env.run(is_training=False)
        for i in range(env.num_players):
            payoffs[i] += earns[i] / iters
    return payoffs


def transform_trajectory(
    trajectories: List[List[int]], payoffs: List[float]
) -> List[List[int | float | bool]]:
    num_players = len(trajectories)
    result: List[List[int | bool | float]] = [[] for _ in range(num_players)]
    for player in range(num_players):
        for i in range(0, len(trajectories[player]) - 2, 2):
            reward: float
            done: bool
            if i == len(trajectories[player]) - 3:
                reward, done = payoffs[player], True
            else:
                reward, done = 0, False
            transition: List[float | bool | int] = copy.deepcopy(
                trajectories[player][i : i + 3]
            )
            transition.insert(2, reward)
            transition.append(done)
            result[player].append(transition)
    return result
