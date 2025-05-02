import jax
import jax.numpy as jnp

from functools import partial

@partial(jax.jit, static_argnames=("nd"))
def fourier_embedding(
    ns: jax.Array, 
    nd: int, 
) -> jax.Array:
    """
    Args:
        ns: input values to embed (vector of floats)
        nd: dimensionality of the embedding (must be even)
    Returns:
        Fourier embedding of shape (ns.len, nd)
    """
    assert nd % 2 == 0, f"nd must be even, got {nd}"
    
    half = nd // 2
    idxs = jnp.arange(half)
    freqs = jnp.exp2(-idxs) * jnp.pi
    
    nsf = jnp.expand_dims(ns, axis=-1)
    x = nsf * freqs
    sin_x = jnp.sin(x)
    cos_x = jnp.cos(x)
    
    return jnp.concatenate([sin_x, cos_x], axis=-1)