import jax
from jax import numpy as jnp
from jaxmarl import make
import random
import pprint
import numpy as np
import argparse

from agents import ManualAgent, LLMAgentBuilder


def get_agents(args, env):
    agents = []
    for player_idx in [0, 1]:
        player_type = getattr(args, f"player{player_idx}")
        if player_type == "manual":
            agents.append(ManualAgent(env, player_idx))
        elif player_type == "llm":
            model_name = getattr(args, f"model{player_idx}")
            version = getattr(args, f"version{player_idx}")
            agents.append(LLMAgentBuilder(env, player_idx, model_name).build(version))
    return agents


def play_game(args):
    print("Starting Hanabi. Remember, actions encoding is:")
    env = make("hanabi")
    pprint.pprint(env.action_encoding)

    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 10000)
    print(f"{'-'*10}\nStarting new game with random seed: {seed}\n")

    agents = get_agents(args, env)

    use_jit = args.use_jit if args.use_jit is not None else True
    with jax.disable_jit(not use_jit):

        rng = jax.random.PRNGKey(seed)
        rng, _rng = jax.random.split(rng)

        prev_state = None
        prev_action = None
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
                agents[i].act(obs, env_state, legal_moves, curr_player, prev_state, prev_action)
                for i in range(len(env.agents))
            ]

            actions = actions_all[curr_player]
            prev_action = int(actions[curr_player])
            print(f"Action played: {env.action_encoding[int(actions[curr_player])]}")

            actions = {
                agent: jnp.array(actions[i]) for i, agent in enumerate(env.agents)
            }

            prev_state = env_state
            rng, env_state, obs, reward, dones, legal_moves = _step_env(
                rng, env_state, actions
            )

            done = dones["__all__"]
            cum_rew += reward["__all__"]
            t += 1

            print("\n" + "=" * 40 + "\n")

        print("Game Ended. Score:", cum_rew)


def main(args):
    play_game(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player0", type=str, default="llm")
    parser.add_argument("--player1", type=str, default="llm")
    parser.add_argument("--model0", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--model1", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--version0", type=str, default="simple")
    parser.add_argument("--version1", type=str, default="simple")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_jit", type=bool, default=True)
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
