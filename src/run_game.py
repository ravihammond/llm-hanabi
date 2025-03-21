import jax
from jax import numpy as jnp
from jaxmarl import make
import random
import pprint
import numpy as np
import argparse
from pprint import pprint

from agents import AgentBuilder


def main(args):
    run_game(args)


def run_game(args):
    pprint(vars(args))

    args = fix_args(args)

    print("Starting Hanabi. Remember, actions encoding is:")
    env = make("hanabi")
    pprint(env.action_encoding)

    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 10000)
    print(f"{'-'*10}\nStarting new game with random seed: {seed}\n")

    agents = get_agents(args, env)

    working_memory = [None, None]
    episodic_memory = [None, None]

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

            episodic_memory = [agents[0].action_history, agents[1].action_history]
            working_memory = [agents[0].working_memory, agents[1].working_memory]

            actions_all = [
                agents[i].act(
                    obs,
                    env_state,
                    legal_moves,
                    curr_player,
                    prev_state,
                    prev_action,
                    episodic_memory,
                    working_memory,
                )
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


def fix_args(args):
    if args.agent != "None":
        args.agent0 = args.agent
        args.agent1 = args.agent
    if args.model != "None":
        args.model0 = args.model
        args.model1 = args.model
    return args


def get_agents(args, env):
    agents = []
    for player_idx in [0, 1]:
        agent_type = getattr(args, f"agent{player_idx}")
        model_name = getattr(args, f"model{player_idx}")
        agents.append(AgentBuilder(
            env, player_idx, model_name, args.verbose
        ).build(agent_type))

    return agents


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="None")
    parser.add_argument("--agent0", type=str, default="llm_simple")
    parser.add_argument("--agent1", type=str, default="llm_simple")
    parser.add_argument("--model", type=str, default="None")
    parser.add_argument("--model0", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--model1", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_jit", type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
