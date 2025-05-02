import jax
import jax.numpy as jnp

from typing import NamedTuple

from .initialize import random_weight
from .layer_norm import layer_norm_rms

class Scale(NamedTuple):
    weights: jax.Array
    bias: jax.Array

def new_scale(key: jax.Array, nstate: int) -> Scale:
    weights = random_weight(key, nstate, 1)
    bias = jnp.asarray(0.0)
    return Scale(weights=weights, bias=bias)

@jax.jit
def scale_forward(scale: Scale, x: jax.Array) -> jax.Array:
    return jnp.exp(layer_norm_rms(x) @ scale.weights + scale.bias)