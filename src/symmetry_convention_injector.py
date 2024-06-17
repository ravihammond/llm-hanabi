import jax
from jax import numpy as jnp
from jaxmarl import make


@jax.jit
def symmetry_convention(state, action, override, convention):
    if not override and len(convention) > 0:
        return action

    print("action before override", action)
    print("overriding action")

    last_action = state.last_action
    print("last_action:", last_action)
    signal_move = convention[0]
    print("signal_move:", signal_move)
    response_move = convention[1]
    print("response_move:", response_move)
    
    return action

