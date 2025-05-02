import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax

from dataclasses import dataclass
import functools
from functools import partial

from typing import NamedTuple

from .transformer import RawTransformer, random_raw_transformer, raw_transformer_forward
from .layer_norm import RMSNorm, new_rms_norm, rms_norm
from .mask import full_mask
from .initialize import random_weight, random_tok_embedding, random_uniform
from .scale import Scale, new_scale, apply_scale
from .embedding import fourier_embedding

@partial(jax.tree_util.register_dataclass,
                   data_fields=['tok_emb', 'row_emb_proj', 'col_emb_proj', 'depth_emb_proj', 'transformer', 'tok_proj', 'value_proj', 'gradient_scale', 'entropy_scale', 'norm_output'],
                   meta_fields=['nstate', 'width', 'depth', 'nvocab'])
@dataclass
class MesaTokenImageTransformer(object):
    tok_emb: jax.Array          # [nvocab, nstate]
    row_emb_proj: jax.Array     # [nstate, nstate]
    col_emb_proj: jax.Array     # [nstate, nstate]
    depth_emb_proj: jax.Array   # [nstate, nstate]
    transformer: RawTransformer
    tok_proj: jax.Array         # [nstate, nvocab]
    value_proj: jax.Array       # [nstate, 1]
    gradient_scale: Scale
    entropy_scale: Scale
    norm_output: RMSNorm
    nstate: int
    width: int
    depth: int
    nvocab: int


def random_mesa_token_image_transformer(key: jax.Array, nvocab: int, nstate: int, n_q_head: int, n_kv_head: int, mult: int, width: int, depth: int, n_blocks: int) -> TokenImageTransformer:
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
    transformer = random_raw_transformer(key_transformer, nstate, n_q_head, n_kv_head, mult, n_blocks)

    key, key_tok_proj = random.split(key)
    tok_proj = random_weight(key_tok_proj, nstate, nvocab)

    key, key_tok_projv = random.split(key)
    value_proj = random_weight(key_tok_projv, nstate, 1)

    key, key_grad_scale, key_entropy_scale = random.split(key, 3)
    gradient_scale = new_scale(key_grad_scale, nstate)
    entropy_scale = new_scale(key_entropy_scale, nstate)

    norm_output = new_rms_norm(nstate)

    return MesaTokenImageTransformer(
        tok_emb=tok_emb,
        row_emb_proj=row_emb_proj,
        col_emb_proj=col_emb_proj,
        depth_emb_proj=depth_emb_proj,
        transformer=transformer,
        tok_proj=tok_proj,
        value_proj=value_proj,
        gradient_scale=gradient_scale,
        entropy_scale=entropy_scale,
        norm_output=norm_output,
        nstate=nstate,
        width=width,
        depth=depth,
        nvocab=nvocab
    )

@jax.jit
def mesa_token_image_transformer_forward_embedding(
    t: MesaTokenImageTransformer,
    tokens: jax.Array,  # [depth, width, width]
) -> jax.Array:         # [ctx_len, nstate]
    depth, width, _ = tokens.shape
    ctx_len = depth * width * width

    tok_embs = t.tok_emb[tokens]  # [depth, width, width, nstate]
    
    # position embeddings
    col_pos_embs = fourier_embedding(jnp.arange(width), t.nstate).astype(tok_embs.dtype) @ t.col_emb_proj                                       # [width, nstate]
    row_pos_embs = fourier_embedding(jnp.arange(width), t.nstate).astype(tok_embs.dtype).reshape(width, 1, t.nstate) @ t.row_emb_proj           # [width, 1, nstate]
    depth_pos_embs = fourier_embedding(jnp.arange(depth), t.nstate).astype(tok_embs.dtype).reshape(depth, 1, 1, t.nstate) @ t.depth_emb_proj    # [depth, 1, 1, nstate]
    
    # combined position embedding
    embedding = (tok_embs + col_pos_embs + row_pos_embs + depth_pos_embs) * 0.5  # [depth, width, width, nstate]
    
    # Reshape for transformer
    return embedding.reshape(ctx_len, t.nstate)  # [ctx_len, nstate]

@jax.jit
def mesa_token_image_transformer_forward_value(
    t: MesaTokenImageTransformer,
    input: jax.Array,   # [ctx_len, nstate]
    state: jax.Array,   # [ctx_len, nstate]
) -> jax.Array:         # [ctx_len, nstate]
    ctx_len, _ = input.shape
    mask = full_mask(ctx_len)  # [ctx_len, ctx_len]

    value = jnp.sum(raw_transformer_forward(t.transformer, input, state, mask) @ t.value_proj)

    return value

@jax.jit
def value_gradient_jitted(t: MesaTokenImageTransformer, input: jax.Array, state: jax.Array):
    return jax.grad(mesa_token_image_transformer_forward_value, argnums=2)(t, input, state)

@jax.jit
def mesa_token_image_transformer_forward_intermediate_step(
    t: MesaTokenImageTransformer,
    input: jax.Array,   # [ctx_len, nstate]
    state: jax.Array,   # [ctx_len, nstate]
    key: jax.Array
) -> jax.Array:         # [ctx_len, nstate]
    gradient = value_gradient_jitted(t, input, state)

    entropy = random_uniform(key, state.shape, 1.0)

    scale_grad = apply_scale(t.gradient_scale, state)
    scale_entropy = apply_scale(t.entropy_scale, state)

    return state + gradient * scale_grad + entropy * scale_entropy

@partial(jax.jit, static_argnames=("nrecur",))
def mesa_token_image_transformer_forward_intermediate(
    t: MesaTokenImageTransformer,
    input: jax.Array,   # [ctx_len, nstate]
    state: jax.Array,   # [ctx_len, nstate]
    nrecur: int, 
    key: jax.Array
) -> jax.Array:         # [ctx_len, nstate]
    for _ in range(nrecur):
        key, key_step = random.split(key)
        state = mesa_token_image_transformer_forward_intermediate_step(t, input, state, key_step)

    return state


@jax.jit
def mesa_token_image_transformer_forward_output(
    t: MesaTokenImageTransformer,
    state: jax.Array,   # [ctx_len, ctx_len]
) -> jax.Array:         # [depth, width, width, nvocab]
    state = rms_norm(t.norm_output, state)
    
    # Project to vocabulary logits
    logits = jnp.matmul(state, t.tok_proj)  # [ctx_len, nvocab]
    
    return logits.reshape(t.depth, t.width, t.width, -1)  # [depth, width, width, nvocab]

@partial(jax.jit, static_argnames=("nrecur",))
def mesa_token_image_transformer_forward(
    t: MesaTokenImageTransformer,
    tokens: jax.Array,  # [depth, width, width]
    nrecur: int, 
    key: jax.Array
) -> jax.Array:         # [depth, width, width, nvocab]
    embedded = mesa_token_image_transformer_forward_embedding(t, tokens)
    latent = mesa_token_image_transformer_forward_intermediate(t, embedded, embedded, nrecur, key)
    return mesa_token_image_transformer_forward_output(t, latent)

@partial(jax.jit, static_argnames=("nrecur",))
def mesa_token_image_transformer_forward_flattened(
    t: MesaTokenImageTransformer,
    tokens: jax.Array,  # [n_ctx]
    nrecur: int, 
    key: jax.Array
) -> jax.Array:         # [n_ctx, nvocab]
    tokens = tokens.reshape(t.depth, t.width, t.width)
    return mesa_token_image_transformer_forward(t, tokens, nrecur, key).reshape(t.depth * t.width * t.width, t.nvocab)