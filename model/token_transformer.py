import jax
import jax.numpy as jnp
import jax.random as random

from typing import Tuple
from dataclasses import dataclass
import functools
from functools import partial

from .transformer import Transformer, random_transformer, transformer_forward_steps, transformer_forward_norm
from .mask import full_mask, causal_mask, io_mask, io_thought_mask
from .initialize import random_weight, random_tok_embedding
from .embedding import fourier_embedding

@partial(jax.tree_util.register_dataclass,
                   data_fields=['tok_emb', 'pos_emb_proj', 'transformer', 'tok_proj'],
                   meta_fields=['nstate', 'nvocab'])
@dataclass
class TokenTransformer(object):
    tok_emb: jax.Array          # [nvocab, nstate]
    pos_emb_proj: jax.Array     # [nstate, nstate]
    transformer: Transformer
    tok_proj: jax.Array         # [nstate, nvocab]
    nstate: int
    nvocab: int

def random_token_transformer(key: jax.Array, nvocab: int, ctx_len: int, nstate: int, n_q_head: int, n_kv_head: int, mult: int, n_blocks: int) -> TokenTransformer:
    assert nstate % n_q_head == 0, "nhead must divide nstate evenly"
    assert (nstate // n_q_head) % 2 == 0, "The head dimension must be even"
    assert n_q_head % n_kv_head == 0, "n_kv_head must divide n_q_head evenly"

    key, key_emb = random.split(key)
    tok_emb = random_tok_embedding(key_emb, nvocab, nstate)

    key, key_pos_emb_proj = random.split(key)
    pos_emb_proj = random_weight(key_pos_emb_proj, nstate, nstate)

    key, key_transformer = random.split(key)
    transformer = random_transformer(key_transformer, nstate, n_q_head, n_kv_head, mult, ctx_len, n_blocks)

    key, key_tok_proj = random.split(key)
    tok_proj = random_weight(key_tok_proj, nstate, nvocab)

    return TokenTransformer(
        tok_emb=tok_emb,
        pos_emb_proj=pos_emb_proj,
        transformer=transformer,
        tok_proj=tok_proj,
        nstate=nstate,
        nvocab=nvocab
    )

@jax.jit
def token_transformer_forward_embedding(
    t: TokenTransformer,
    tokens: jax.Array,  # [ctx_len]
) -> jax.Array:         # [ctx_len, nstate]
    ctx_len = tokens.shape
    
    tok_embs = t.tok_emb[tokens] # [ctx_len, nstate]
    pos_embs = fourier_embedding(jnp.arange(ctx_len), t.nstate).astype(tok_embs.dtype) @ t.pos_emb_proj # [ctx_len, nstate]

    scale = 2.0 ** -0.5
    embedding = (tok_embs + pos_embs) * scale

    return embedding

@partial(jax.jit, static_argnames=("nrecur",))
def token_transformer_forward_intermediate(
    t: TokenTransformer,
    input: jax.Array,   # [ctx_len, nstate]
    state: jax.Array,   # [ctx_len, nstate]
    mask: jax.Array,    # [ctx_len, ctx_len]
    nrecur: int, 
    key: jax.Array
) -> jax.Array:         # [ctx_len, nstate]
    return transformer_forward_steps(t.transformer, input, state, mask, nrecur, key)


@jax.jit
def token_transformer_forward_output(
    t: TokenTransformer,
    state: jax.Array,   # [ctx_len, ctx_len]
) -> jax.Array:         # [ctx_len, nvocab]
    normed = transformer_forward_norm(t.transformer, state)
    
    # Project to vocabulary logits
    logits = normed @ t.tok_proj  # [ctx_len, nvocab]
    return logits

@partial(jax.jit, static_argnames=("nrecur",))
def token_transformer_forward(
    t: TokenTransformer,
    tokens: jax.Array,  # [ctx_len]
    mask: jax.Array,    # [ctx_len, ctx_len]
    nrecur: int, 
    key: jax.Array
) -> jax.Array:         # [ctx_len, nvocab]
    embedded = token_transformer_forward_embedding(t, tokens)
    latent = token_transformer_forward_intermediate(t, embedded, embedded, mask, nrecur, key)
    return token_transformer_forward_output(t, latent)

@partial(jax.jit, static_argnames=("nrecur",))
def token_transformer_forward_autoregressive(
    t: TokenTransformer,
    tokens: jax.Array,  # [ctx_len]
    nrecur: int, 
    key: jax.Array
) -> jax.Array:         # [ctx_len, nvocab]
    ctx_len = tokens.shape
    mask = causal_mask(ctx_len)
    return token_transformer_forward(t, tokens, mask, nrecur, key)

@partial(jax.jit, static_argnames=("nrecur",))
def token_transformer_forward_autoregressive_io(
    t: TokenTransformer,
    tokens_input: jax.Array,    # [ctx_len_input]
    tokens_output: jax.Array,   # [ctx_len_output]
    nrecur: int, 
    key: jax.Array
) -> Tuple[jax.Array, jax.Array]:                 # ([ctx_len_input, nvocab], [ctx_len_output, nvocab])
    ctx_len_input, ctx_len_output = tokens_input.shape, tokens_output.shape
    mask = io_mask(ctx_len_input, ctx_len_output)
    tokens = jnp.append(tokens_input, tokens_output)

    logits = token_transformer_forward(t, tokens, mask, nrecur, key)
    logits_input = logits[:ctx_len_input, :]
    logits_output = logits[ctx_len_input:, :]

    return (logits_input, logits_output)

@partial(jax.jit, static_argnames=("nrecur",))
def token_transformer_forward_autoregressive_io_thought(
    t: TokenTransformer,
    tokens_input: jax.Array,                    # [ctx_len_input]
    tokens_output: jax.Array,                   # [ctx_len_output]
    n_thought_tokens: int,
    thought_token: int, 
    nrecur: int, 
    key: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:    # ([ctx_len_input, nvocab], [n_thought_tokens, nvocab], [ctx_len_output, nvocab])
    ctx_len_input, ctx_len_output = tokens_input.shape, tokens_output.shape
    mask = io_thought_mask(ctx_len_input, n_thought_tokens, ctx_len_output)

    thought_tokens = jnp.full(n_thought_tokens, thought_token)
    tokens = jnp.append(tokens_input, thought_tokens, tokens_output)

    thought_end = ctx_len_input + n_thought_tokens
    output_start = thought_end

    logits = token_transformer_forward(t, tokens, mask, nrecur, key)
    logits_input = logits[:ctx_len_input, :]
    logits_thought = logits[ctx_len_input:thought_end, :]
    logits_output = logits[output_start:, :]

    return (logits_input, logits_thought, logits_output)