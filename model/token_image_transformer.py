import jax
import jax.numpy as jnp
import jax.random as random

from dataclasses import dataclass
import functools
from functools import partial

from .transformer import Transformer, random_transformer, transformer_forward_steps, transformer_forward_norm
from .mask import full_mask
from .initialize import random_weight, random_tok_embedding
from .embedding import fourier_embedding

@partial(jax.tree_util.register_dataclass,
                   data_fields=['tok_emb', 'row_emb_proj', 'col_emb_proj', 'depth_emb_proj', 'transformer', 'tok_proj'],
                   meta_fields=['nstate', 'width', 'depth', 'nvocab'])
@dataclass
class TokenImageTransformer(object):
    tok_emb: jax.Array          # [nvocab, nstate]
    row_emb_proj: jax.Array     # [nstate, nstate]
    col_emb_proj: jax.Array     # [nstate, nstate]
    depth_emb_proj: jax.Array   # [nstate, nstate]
    transformer: Transformer
    tok_proj: jax.Array         # [nstate, nvocab]
    nstate: int
    width: int
    depth: int
    nvocab: int
    


def random_token_image_transformer(key: jax.Array, nvocab: int, nstate: int, n_q_head: int, n_kv_head: int, mult: int, width: int, depth: int, n_blocks: int) -> TokenImageTransformer:
    assert nstate % n_q_head == 0, "nhead must divide nstate evenly"
    assert (nstate // n_q_head) % 2 == 0, "The head dimension must be even"
    assert n_q_head % n_kv_head == 0, "n_kv_head must divide n_q_head evenly"

    key, key_emb = random.split(key)
    tok_emb = random_tok_embedding(key_emb, nvocab, nstate)

    key, key_row_proj, key_col_proj, key_depth_proj = random.split(key, 4)
    row_emb_proj = random_weight(key_row_proj, nstate, nstate)
    col_emb_proj = random_weight(key_col_proj, nstate, nstate)
    depth_emb_proj = random_weight(key_depth_proj, nstate, nstate)

    ctx_len = width * width * depth
    key, key_transformer = random.split(key)
    transformer = random_transformer(key_transformer, nstate, n_q_head, n_kv_head, mult, ctx_len, n_blocks)

    key, key_tok_proj = random.split(key)
    tok_proj = random_weight(key_tok_proj,nstate, nvocab)

    return TokenImageTransformer(tok_emb, row_emb_proj, col_emb_proj, depth_emb_proj, transformer, tok_proj, nstate, width, depth, nvocab)

@jax.jit
def token_image_transformer_forward_embedding(
    t: TokenImageTransformer,
    tokens: jax.Array,  # [depth, width, width]
) -> jax.Array:         # [ctx_len, nstate]
    depth, width, _ = tokens.shape
    ctx_len = depth * width * width
    
    tok_embs = t.tok_emb[tokens] # [depth, width, width, nstate]
    
    # position embeddings
    col_pos_embs = fourier_embedding(jnp.arange(width), t.nstate).astype(tok_embs.dtype) @ t.col_emb_proj                                      # [width, nstate]
    row_pos_embs = fourier_embedding(jnp.arange(width), t.nstate).astype(tok_embs.dtype).reshape(width, 1, t.nstate) @ t.row_emb_proj          # [width, 1, nstate]
    depth_pos_embs = fourier_embedding(jnp.arange(depth), t.nstate).astype(tok_embs.dtype).reshape(depth, 1, 1, t.nstate) @ t.depth_emb_proj   # [depth, 1, 1, nstate]
    
    # combined position embedding
    embedding = (tok_embs + col_pos_embs + row_pos_embs + depth_pos_embs) * 0.5 # [depth, width, width, nstate]
    
    # Reshape for transformer
    return embedding.reshape(ctx_len, t.nstate) # [ctx_len, nstate]

@partial(jax.jit, static_argnames=("nrecur",))
def token_image_transformer_forward_intermediate(
    t: TokenImageTransformer,
    input: jax.Array,   # [ctx_len, nstate]
    state: jax.Array,   # [ctx_len, nstate]
    nrecur: int, 
    key: jax.Array
) -> jax.Array:         # [ctx_len, nstate]
    ctx_len, _ = input.shape
    mask = full_mask(ctx_len) # [ctx_len, ctx_len]

    return transformer_forward_steps(t.transformer, input, state, mask, nrecur, key)


@jax.jit
def token_image_transformer_forward_output(
    t: TokenImageTransformer,
    state: jax.Array,   # [ctx_len, ctx_len]
) -> jax.Array:         # [depth, width, width, nvocab]
    normed = transformer_forward_norm(t.transformer, state)
    
    # Project to vocabulary logits
    logits = normed @ t.tok_proj  # [ctx_len, nvocab]
    
    return logits.reshape(t.depth, t.width, t.width, -1)  # [depth, width, width, nvocab]

@partial(jax.jit, static_argnames=("nrecur",))
def token_image_transformer_forward(
    t: TokenImageTransformer,
    tokens: jax.Array,  # [depth, width, width]
    nrecur: int, 
    key: jax.Array
) -> jax.Array:         # [depth, width, width, nvocab]
    embedded = token_image_transformer_forward_embedding(t, tokens)
    latent = token_image_transformer_forward_intermediate(t, embedded, embedded, nrecur, key)
    return token_image_transformer_forward_output(t, latent)

@partial(jax.jit, static_argnames=("nrecur",))
def token_image_transformer_forward_flattened(
    t: TokenImageTransformer,
    tokens: jax.Array,  # [n_ctx]
    nrecur: int, 
    key: jax.Array
) -> jax.Array:         # [n_ctx, nvocab]
    tokens = tokens.reshape(t.depth, t.width, t.width)
    return token_image_transformer_forward(t, tokens, nrecur, key).reshape(t.depth * t.width * t.width, t.nvocab)