import jax
import jax.numpy as jnp
import jax.random as random

from dataclasses import dataclass
from functools import partial

from .initialize import random_weight
from .mask import apply_mask

@partial(jax.tree_util.register_dataclass,
                   data_fields=['q', 'k', 'v', 'proj'],
                   meta_fields=['nstate', 'n_q_head', 'n_kv_head'])
@dataclass
class AttentionBlock(object):
    nstate: int
    n_q_head: int   # Must divide nstate evenly and nstate/n_q_head must be even for rotary embeddings
    n_kv_head: int  # Must divide n_q_head evenly
    q: jax.Array    # [nstate, nstate]
    k: jax.Array    # [nstate, nstate/n_q_head*n_kv_head]
    v: jax.Array    # [nstate, nstate/n_q_head*n_kv_head]
    proj: jax.Array # [nstate, nstate]

def random_attention_block(key: jax.Array, nstate: int, n_q_head: int, n_kv_head: int) -> AttentionBlock:
    assert nstate % n_q_head == 0, "n_q_head must divide nstate evenly"
    assert (nstate // n_q_head ) % 2 == 0, "The head dimension must be even"
    assert n_q_head % n_kv_head == 0, "n_kv_head must divide n_q_head evenly"

    keyq, keyk, keyv, keyproj = random.split(key, num=4)

    kv_size = nstate // n_q_head * n_kv_head
    q = random_weight(keyq, nstate, nstate)
    k = random_weight(keyk, nstate, kv_size)
    v = random_weight(keyv, nstate, kv_size)
    proj = random_weight(keyproj, nstate, nstate)

    return AttentionBlock(nstate=nstate, n_q_head=n_q_head, n_kv_head=n_kv_head, q=q, k=k, v=v, proj=proj)

@partial(jax.jit, static_argnames=("n_rep"))
def repeat_kv_interleaved(x: jax.Array, n_rep: int) -> jax.Array:
    seq_len, n_head, nstate = x.shape
    return jnp.repeat(x, n_rep, axis=-2)


@jax.jit
def attention(
    ab: AttentionBlock,
    x: jax.Array,  # [seq_len, nstate]
    mask: jax.Array # [seq_len, seq_len]
) -> jax.Array:  # [seq_len, nstate]
    """Compute causal grouped query attention block."""
    seq_len = x.shape[0]
    head_dim = ab.nstate // ab.n_q_head
    
    q = x @ ab.q # [seq_len, nstate]
    k = x @ ab.k # [seq_len, nstate/n_q_head*n_kv_head]
    v = x @ ab.v # [seq_len, nstate/n_q_head*n_kv_head]

    #qkv = x @ ab.qkv
    #q, k, v = jnp.split(qkv, 3, axis=-1)
    
    # Reshape to heads [n_q_head, seq_len, head_dim]
    q = q.reshape(seq_len, ab.n_q_head, head_dim).transpose(1, 0, 2)
    n_rep = ab.n_q_head // ab.n_kv_head
    k = repeat_kv_interleaved(k.reshape(seq_len, ab.n_kv_head, head_dim), n_rep).transpose(1, 0, 2)
    v = repeat_kv_interleaved(v.reshape(seq_len, ab.n_kv_head, head_dim), n_rep).transpose(1, 0, 2)
    
    # Compute attention scores [nhead, seq_len, seq_len]
    scores = (q @ jnp.swapaxes(k, -2, -1)) / jnp.sqrt(head_dim)
    scores = apply_mask(mask, scores)
    
    # Apply attention
    weights = jax.nn.softmax(scores, axis=-1)
    attended = weights @ v  # [nhead, seq_len, head_dim]
    
    output = attended.transpose(1, 0, 2).reshape(seq_len, ab.nstate)
    return output @ ab.proj
