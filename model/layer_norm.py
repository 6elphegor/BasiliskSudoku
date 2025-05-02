import jax
import jax.numpy as jnp

from typing import NamedTuple

@jax.jit
def layer_norm_rms(x: jax.Array, eps: float = 1e-5) -> jax.Array:
    """ Apply RMS normalization. 
        'Tis essentially a projection to a point on a hypersphere,
          scaled by the square root of the number of elements.
        Important, because standard normal distributed values are roughly fixed points.
    """
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms

class RMSNorm(NamedTuple):
    weights: jax.Array

def new_rms_norm(nstate: int) -> RMSNorm:
    weights = jnp.zeros((nstate,))
    return RMSNorm(weights=weights)

@jax.jit
def rms_norm(rn: RMSNorm, x: jax.Array) -> jax.Array:
    return layer_norm_rms(x) * (rn.weights + 1) # default weights are zero