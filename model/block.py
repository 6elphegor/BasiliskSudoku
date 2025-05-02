import jax
import jax.numpy as jnp
import jax.random as random

from typing import NamedTuple
from functools import partial

from .attention import AttentionBlock, attention, random_attention_block
from .feedforward import FeedforwardLlamaBlock, feedforward_llama, random_feedforward_llama_block
from .layer_norm import RMSNorm, rms_norm, new_rms_norm

class Block(NamedTuple):
    ab: AttentionBlock
    ff: FeedforwardLlamaBlock
    norm: RMSNorm

def random_block(key: jax.Array, nstate: int, n_q_head: int, n_kv_head: int, mult: int) -> Block:
    assert nstate % n_q_head == 0, "nhead must divide nstate evenly"
    assert (nstate // n_q_head) % 2 == 0, "The head dimension must be even"
    assert n_q_head % n_kv_head == 0, "n_kv_head must divide n_q_head evenly"

    key1, key2 = random.split(key)
    ab = random_attention_block(key1, nstate, n_q_head, n_kv_head)
    ff = random_feedforward_llama_block(key2, nstate, mult)
    norm = new_rms_norm(nstate)

    return Block(
        ab=ab,
        ff=ff,
        norm=norm
    )


@jax.jit
def block_forward(b: Block, x: jax.Array, mask: jax.Array, scale: float = 1.0) -> jax.Array:
    """Apply a full transformer block."""
    x = x + attention(b.ab, rms_norm(b.norm, x), mask) * scale
    return x + feedforward_llama(b.ff, rms_norm(b.norm, x)) * scale
