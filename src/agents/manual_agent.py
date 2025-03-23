import numpy as np
import sys

from agents.agent import Agent

from agents.mappings import ID_TO_ACTION


class ManualAgent(Agent):
    def __init__(self, env, player_idx):
        Agent.__init__(self, env, player_idx)
        self.action_history = []
        self.working_memory = []

    def _act(
        self,
        obs,
        env_state,
        legal_moves,
        curr_player,
        env,
        prev_state,
        prev_action,
        episodic_memory,
        working_memory
    ):
        actions = np.array([20, 20])

        print("valid actions:", [(a, self._env.action_encoding[a]) for a in np.where(legal_moves[curr_player] == 1)[0]])

        if curr_player != self._player_idx:
            return actions

        # take action input from user
        while True:
            try:
                print("---")
                action = int(input("Insert manual action: "))
                print("action legal:", legal_moves[curr_player][action])
                print("---\n")
                if (
                    action >= 0
                    and action <= 20
                    and legal_moves[curr_player][action] == 1
                ):
                    break
                else:
                    print("Invalid action.")
            except KeyboardInterrupt:
                sys.exit(0)
            except:
                action = 0
                print("Invalid action.")

        actions[curr_player] = action
        self.action_history.append(ID_TO_ACTION[action])

        return actions
