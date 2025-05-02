import jax
import jax.numpy as jnp

from functools import partial

@partial(jax.jit, static_argnames=("seq_len"))
def full_mask(seq_len: int) -> jax.Array:
    return jnp.ones((seq_len, seq_len), dtype=jnp.bool_)

@partial(jax.jit, static_argnames=("seq_len"))
def causal_mask(seq_len: int) -> jax.Array:
    return jnp.tril(full_mask(seq_len))

@partial(jax.jit, static_argnames=("n_input_toks", "n_output_toks"))
def io_mask(n_input_toks: int, n_output_toks: int) -> jax.Array:
    total_toks = n_input_toks + n_output_toks
    mask = causal_mask(total_toks)
    return mask.at[:n_input_toks, :n_input_toks].set(True)

@partial(jax.jit, static_argnames=("n_input_toks", "n_thought_toks", "n_output_toks"))
def io_thought_mask(n_input_toks: int, n_thought_toks: int, n_output_toks: int) -> jax.Array:
    total_toks = n_input_toks + n_thought_toks + n_output_toks
    thought_start, thought_end = n_input_toks, n_input_toks + n_thought_toks
    mask = causal_mask(total_toks)
    return mask.at[thought_start:thought_end, :thought_end].set(True)

@jax.jit
def apply_mask(mask: jax.Array, x: jax.Array) -> jax.Array:
    return jnp.where(mask, x, -jnp.inf)