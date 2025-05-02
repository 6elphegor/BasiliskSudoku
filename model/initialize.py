import jax
import jax.numpy as jnp
from jax import random

import functools
from functools import partial

@partial(jax.jit, static_argnames=("shape",))
def random_uniform(key: jax.Array, shape, stddev: float) -> jax.Array:
    """Generates uniformly distributed values with mean 0 and standard deviation stddev"""
    limit = jnp.sqrt(3.0) * stddev
    return random.uniform(key, shape, minval=-limit, maxval=limit)

@partial(jax.jit, static_argnames=("n_in", "n_out",))
def random_weight(key: jax.Array, n_in: int, n_out: int) -> jax.Array:
    """Preserves element-wise stddev of 1 when applied to a vector, v @ W"""
    limit = jnp.sqrt(3.0 / n_in)
    return random.uniform(key, (n_in, n_out), minval=-limit, maxval=limit)

@partial(jax.jit, static_argnames=("nvocab", "nstate"))
def random_tok_embedding(key: jax.Array, nvocab: int, nstate: int) -> jax.Array:
    return random_uniform(key, (nvocab, nstate), 1.0)