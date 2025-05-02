import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax

from typing import NamedTuple

from dataclasses import dataclass
import functools
from functools import partial

from .block import Block, random_block, block_forward
from .layer_norm import RMSNorm, new_rms_norm, rms_norm
from .initialize import random_uniform, random_tok_embedding
from .scale import Scale, new_scale, scale_forward

class RawTransformer(NamedTuple):
    blocks: list[Block]
    norm_input: RMSNorm
    norm_output: RMSNorm

def random_raw_transformer(key: jax.Array, nstate: int, n_q_head: int, n_kv_head: int, mult: int, n_blocks: int) -> RawTransformer:
    assert nstate % n_q_head == 0, "nhead must divide nstate evenly"
    assert (nstate // n_q_head) % 2 == 0, "The head dimension must be even"
    assert n_q_head % n_kv_head == 0, "n_kv_head must divide n_q_head evenly"
    
    block_keys = random.split(key, n_blocks)
    blocks = [random_block(block_key, nstate, n_q_head, n_kv_head, mult) for block_key in block_keys]

    norm_input = new_rms_norm(nstate)
    norm_output = new_rms_norm(nstate)
    
    return RawTransformer(blocks, norm_input, norm_output)

@jax.jit
def raw_transformer_forward(
    t: RawTransformer,
    input: jax.Array,       # [seq_len, nstate]
    state: jax.Array,       # [seq_len, nstate]
    mask: jax.Array
) -> jax.Array:             # [seq_len, nstate]
    scale = (2 * len(t.blocks)) ** -0.5

    state = rms_norm(t.norm_input, state)
    state = state + input

    for block in t.blocks:
        state = block_forward(block, state, mask, scale)

    state = rms_norm(t.norm_output, state)

    return state

@partial(jax.tree_util.register_dataclass,
                   data_fields=['blocks', 'norm_input', 'norm_output', 'local_state_scale', 'entropy_scale', 'input_scale'],
                   meta_fields=['nstate', 'n_q_head', 'n_kv_head', 'mult', 'maxctx'])
@dataclass
class Transformer(object):
    nstate: int
    n_q_head: int               # Must divide nstate evenly and nstate/n_q_head must be even for rotary embeddings
    n_kv_head: int              # Must divide n_q_head evenly
    mult: int                   # Feedforward hidden state size = mult * nstate
    maxctx: int                 # Bound context length because may not generalize outside of training bounds
    blocks: list[Block]
    norm_input: RMSNorm         # Normalizes the global state
    norm_output: RMSNorm        # Normalizes the final output
    local_state_scale: Scale    # Scales the local state before adding it to the global state
    entropy_scale: Scale        # Scales the entropy before adding it to the global state
    input_scale: jax.Array      # Scales the input before adding it to the local state


def random_transformer(key: jax.Array, nstate: int, n_q_head: int, n_kv_head: int, mult: int, maxctx: int, n_blocks: int) -> Transformer:
    assert nstate % n_q_head == 0, "nhead must divide nstate evenly"
    assert (nstate // n_q_head) % 2 == 0, "The head dimension must be even"
    assert n_q_head % n_kv_head == 0, "n_kv_head must divide n_q_head evenly"

    block_keys = random.split(key, n_blocks)
    blocks = [random_block(block_key, nstate, n_q_head, n_kv_head, mult) for block_key in block_keys]

    norm_input = new_rms_norm(nstate)
    norm_output = new_rms_norm(nstate)

    key, key_scale_loc = random.split(key)
    local_state_scale = new_scale(key_scale_loc, nstate)

    key, key_scale_entropy = random.split(key)
    entropy_scale = new_scale(key_scale_entropy, nstate)

    input_scale = jnp.asarray(0.0)
    
    return Transformer(
        nstate=nstate,
        n_q_head=n_q_head,
        n_kv_head=n_kv_head,
        mult=mult,
        maxctx=maxctx,
        blocks=blocks,
        norm_input=norm_input,
        norm_output=norm_output,
        local_state_scale=local_state_scale,
        entropy_scale=entropy_scale,
        input_scale=input_scale
    )

@jax.jit
def transformer_forward_step(
    t: Transformer,
    input: jax.Array,   # [seq_len, nstate]
    state: jax.Array,   # [seq_len, nstate]
    mask: jax.Array,
    key: jax.Array
) -> jax.Array:         # [seq_len, nstate]
    """Apply transformer without normalizing output"""
    seq_len = input.shape[0]
    head_state_size = t.nstate // t.n_q_head
    assert seq_len <= t.maxctx, f"Sequence length {seq_len} exceeds maximum context {t.maxctx}"

    state_local = rms_norm(t.norm_input, state)
    state_local = state_local + input * jnp.exp(t.input_scale)

    # Scale so that the blocks overall add in roughly standard normal distributed values
    # This helps pass the token embedding information throughout all the layers
    scale = (2 * len(t.blocks)) ** -0.5

    for block in t.blocks:
        state_local = block_forward(block, state_local, mask, scale)

    entropy = random_uniform(key, state_local.shape, 1.0)

    local_scale = scale_forward(t.local_state_scale, state_local)
    entropy_scale = scale_forward(t.entropy_scale, state_local)

    return state + state_local * local_scale + entropy * entropy_scale  

@partial(jax.jit, static_argnames=("nrecur"))
def transformer_forward_steps(
    t: Transformer,
    input: jax.Array,   # [seq_len, nstate]
    state: jax.Array,   # [seq_len, nstate]
    mask: jax.Array,
    nrecur: int,  
    key: jax.Array
) -> jax.Array:         # [seq_len, nstate]
    """Apply transformer without normalizing output"""

    for _ in range(nrecur):
        key, key_step = random.split(key)
        state = transformer_forward_step(t, input, state, mask, key_step)
    
    return state

@jax.jit
def transformer_forward_norm(
    t: Transformer,
    state: jax.Array,   # [seq_len, nstate]
) -> jax.Array:         # [seq_len, nstate]
    """Normalize state"""
    return rms_norm(t.norm_output, state)

@partial(jax.jit, static_argnames=("nrecur"))
def transformer_forward(
    t: Transformer,
    input: jax.Array,   # [seq_len, nstate]
    mask: jax.Array,
    nrecur: int, 
    key: jax.Array
) -> jax.Array:         # [seq_len, nstate]
    state = transformer_forward_steps(t, input, input, mask, nrecur, key)
    return transformer_forward_norm(t, state)

