from abc import ABC
import numpy as np
import sys
from jax import numpy as jnp
from pprint import pprint

class Agent(ABC):
    def __init__(self, env, player_idx):
        self._env = env
        self._player_idx = player_idx

    def act(
        self,
        obs,
        env_state,
        legal_moves,
        curr_player,
        prev_state,
        prev_action,
    ):
        obs = batchify(obs, self._env)
        legal_moves = batchify(legal_moves, self._env)
        if curr_player != self._player_idx:
            return np.array([20, 20])
        actions = self._act(
            obs,
            env_state,
            legal_moves,
            curr_player,
            prev_state,
            prev_action,
        )
        return actions

    def _act(
        self,
        obs,
        state,
        legal_moves,
        curr_player,
        prev_state,
        prev_action,
    ):
        return NotImplementedError

def batchify(x, env):
    return jnp.stack([x[agent] for agent in env.agents])

