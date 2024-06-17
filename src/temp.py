import jax
import jax.numpy as jnp
from jax import random

key = random.key(1701)

mat = random.normal(key, (150, 100))
print("mat:", mat.shape)
batched_x = random.normal(key, (10, 100))
print("batched_x:", batched_x.shape)

def apply_matrix(v):
    print("        apply_matrix()")
    print("        mat:", mat.shape)
    print("        v:", v.shape)
    print()
    return jnp.dot(mat, v)

def naively_batched_apply_matrix(v_batched):
    print("    naively_batched_apply_matrix()")
    print("    v_batched:", v_batched.shape)
    print()

    applied_matrices = []
    for v in v_batched:
        print("    v:", v.shape)
        applied_matrix = apply_matrix(v)
        print("    applied_matrix:", applied_matrix.shape)
        applied_matrices.append(applied_matrix)
    return jnp.stack(applied_matrices)

print('Naively batched')
result = naively_batched_apply_matrix(batched_x).block_until_ready()
print("result:", result.shape)

