import jax
import jax.numpy as jnp
from jax import random

from typing import NamedTuple
from functools import partial

from .initialize import random_weight

class FeedforwardBlock(NamedTuple):
    m1: jax.Array  # [nstate, mult * nstate]
    m2: jax.Array  # [mult * nstate, nstate]

@jax.jit
def feedforward(ff: FeedforwardBlock, x: jax.Array) -> jax.Array:
    """Apply feedforward layer with GELU activation."""
    return (jax.nn.gelu(x @ ff.m1)) @ ff.m2

class FeedforwardLlamaBlock(NamedTuple):
    m1: jax.Array  # [nstate, mult * nstate]
    m2: jax.Array  # [mult * nstate, nstate]
    m3: jax.Array  # [nstate, mult * nstate]

def random_feedforward_llama_block(key: jax.Array, nstate: int, mult: int) -> FeedforwardLlamaBlock:
    k1, k2, k3 = random.split(key, num=3)

    n_hidden_state = nstate * mult
    m1 = random_weight(k1, nstate, n_hidden_state)
    m2 = random_weight(k2, n_hidden_state, nstate)
    m3 = random_weight(k3, nstate, n_hidden_state)

    return FeedforwardLlamaBlock(m1=m1, m2=m2, m3=m3)

@jax.jit
def feedforward_llama(ff: FeedforwardLlamaBlock, x: jax.Array) -> jax.Array:
    """Apply llama style feedforward layer with SILU activation."""
    return (jax.nn.silu(x @ ff.m1) * (x @ ff.m3)) @ ff.m2