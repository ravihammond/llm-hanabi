import numpy as np

from agents.base_llm_agent import BaseLLMAgent

class SelfVerifLLMAgent(BaseLLMAgent):
    def __init__(self, player_idx, model_name, env, verbose):
        super().__init__(player_idx, model_name, env, verbose)

    def _act(
        self, 
        obs, 
        state, 
        legal_moves, 
        curr_player, 
        env, 
        prev_state, 
        prev_action
    ):
        actions = np.array([ 20, 20 ])
        return actions
