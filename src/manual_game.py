import jax
from jax import numpy as jnp
from jaxmarl import make
import random
import pprint
import sys
import numpy as np
import argparse
import json

from jaxmarl.wrappers.baselines import load_params
from jaxmarl.environments.hanabi.pretrained.obl_r2d2_agent import OBLAgentR2D2
from symmetry_convention_injector import symmetry_convention

env = make("hanabi")
batchify = lambda x: jnp.stack([x[agent] for agent in env.agents])
unbatchify = lambda x: {agent: x[i] for i, agent in enumerate(env.agents)}

STR_TO_MOVE = {
    'D0': 0,
    'D1': 1, 
    'D2': 2, 
    'D3': 3, 
    'D4': 4, 
    'P0': 5, 
    'P1': 6, 
    'P2': 7, 
    'P3': 8, 
    'P4': 9, 
    'HR': 10, 
    'HY': 11, 
    'HG': 12, 
    'HW': 13, 
    'HB': 14, 
    'H1': 15,
    'H2': 16, 
    'H3': 17, 
    'H4': 18, 
    'H5': 19, 
}


class ManualPlayer:
    def __init__(self, player_idx):
        self._player_idx = player_idx

    def act(self, obs, legal_moves, curr_player) -> int:
        legal_moves = batchify(legal_moves)
        actions = np.array([20, 20])

        print("valid actions:", [(a, env.action_encoding[a]) for a in np.where(legal_moves[curr_player] == 1)[0]])

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

        return actions


class ManualGameOBLAgentR2D2:
    def __init__(self, weight_file, player_idx):
        self.player_id = player_idx
        self.params = load_params(weight_file)
        self.model = OBLAgentR2D2()
        self.carry = self.model.initialize_carry(jax.random.PRNGKey(0), batch_dims=(2,))

    def act(self, obs, legal_moves, curr_player):
        obs = batchify(obs)
        legal_moves = batchify(legal_moves)
        self.carry, actions = self.model.greedy_act(
            self.params, self.carry, (obs, legal_moves)
        )
        return actions


def get_agents(args):
    agents = []
    for player_idx in [0, 1]:
        player_type = getattr(args, f"player{player_idx}")
        weight_file = getattr(args, f"weight{player_idx}")
        if player_type == "manual":
            agents.append(ManualPlayer(player_idx))
        elif player_type == "obl":
            assert (
                weight_file is not None
            ), "Weight file must be provided for all the OBL agents."
            agents.append(ManualGameOBLAgentR2D2(weight_file, player_idx))
    return agents


def play_game(args):
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 10000)
    print(f"{'-'*10}\nStarting new game with random seed: {seed}\n")

    agents = get_agents(args)

    convention = [STR_TO_MOVE[x] for x in load_json_list(args.convention)]

    use_jit = args.use_jit if args.use_jit is not None else True
    with jax.disable_jit(not use_jit):

        rng = jax.random.PRNGKey(seed)
        rng, _rng = jax.random.split(rng)

        obs, env_state = env.reset(_rng)
        legal_moves = env.get_legal_moves(env_state)

        @jax.jit
        def _step_env(rng, env_state, actions):
            rng, _rng = jax.random.split(rng)
            new_obs, new_env_state, reward, dones, infos = env.step(
                _rng, env_state, actions
            )
            new_legal_moves = env.get_legal_moves(new_env_state)
            return rng, new_env_state, new_obs, reward, dones, new_legal_moves

        done = False
        cum_rew = 0
        t = 0

        print("\n" + "=" * 40 + "\n")

        while not done:
            env.render(env_state)

            curr_player = np.where(env_state.cur_player_idx == 1)[0][0]

            actions_all = [
                symmetry_convention(env_state, agents[i].act(obs, legal_moves, curr_player), 
                                    getattr(args, f"convention_override{i}"), convention)
                for i in range(len(env.agents))
            ]

            actions = actions_all[curr_player]
            print(f"\nAction played: {env.action_encoding[int(actions[curr_player])]}")

            actions = {
                agent: jnp.array(actions[i]) for i, agent in enumerate(env.agents)
            }

            rng, env_state, obs, reward, dones, legal_moves = _step_env(
                rng, env_state, actions
            )

            done = dones["__all__"]
            cum_rew += reward["__all__"]
            t += 1

            print("\n" + "=" * 40 + "\n")

        print("Game Ended. Score:", cum_rew)


def load_json_list(path):
    print("load_json_list:", path)
    if path == "None":
        return []
    file = open(path)
    return json.load(file)


def main(args):
    print("Starting Hanabi. Remember, actions encoding is:")
    pprint.pprint(env.action_encoding)
    play_game(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player0", type=str, default="obl")
    parser.add_argument("--player1", type=str, default="obl")
    parser.add_argument("--weight0", type=str, default=None)
    parser.add_argument("--weight1", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_jit", type=bool, default=True)
    parser.add_argument("--convention", type=str, default=None)
    parser.add_argument("--convention_override0", type=int, default=0)
    parser.add_argument("--convention_override1", type=int, default=0)
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
