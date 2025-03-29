from abc import ABC
import numpy as np
import sys
from jax import numpy as jnp
from pprint import pprint

class Agent(ABC):
    def __init__(self, env, player_idx):
        self._env = env
        self._player_idx = player_idx

    def act(self, obs, curr_player, turn, score, legal_moves):
        if curr_player != self._player_idx:
            return len(legal_moves) - 1

        actions = self._act(obs, curr_player, turn, score, legal_moves)

        return actions

    def _act(self, obs, curr_player, turn, score, legal_moves):
        return NotImplementedError

