from agents.manual_agent import ManualAgent
from agents.old_llm_agent import OldLLMAgent
from agents.simple_llm_agent import SimpleLLMAgent
from agents.self_verif_llm_agent import SelfVerifLLMAgent

class AgentBuilder():
    def __init__(self, env, player_idx, model_name, verbose=1) -> None:
        self._env = env
        self._player_idx = player_idx
        self._model_name = model_name
        self._verbose = verbose

    def build(self, version):
        if version == "manual":
            return ManualAgent(
                self._env,
                self._player_idx,
            )
        elif version == "llm_simple":
            return SimpleLLMAgent(
                self._env,
                self._player_idx,
                self._model_name,
                self._verbose
            )
        elif version == "llm_self_verif":
            return SelfVerifLLMAgent(
                self._env,
                self._player_idx,
                self._model_name,
                self._verbose
            )
        elif version == "llm_old":
            return OldLLMAgent(
                self._env,
                self._player_idx,
                self._model_name,
                self._verbose
            )
        else:
            print("Error: unknown agent version:", version)
            exit()
