import jax
import jax.numpy as jnp
from functools import partial
from typing import Any, Callable, Tuple

from optimizer import AdamW, update

@jax.jit
def cross_entropy_loss(tokens: jax.Array, logits: jax.Array, mask: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_log_probs = jnp.take_along_axis(log_probs, tokens[..., None], axis=-1).squeeze(-1)
    loss = -jnp.sum(jnp.where(mask, token_log_probs, 0.0)) / (jnp.sum(mask) + 1e-8)
    return loss

@partial(jax.jit, static_argnames=("forward", "nrecur"))
def autoregressive_loss(model: Any, forward: Callable[[Any, jax.Array, int], jax.Array], tokens: jax.Array, mask: jax.Array, nrecur: int, key: jax.Array) -> jax.Array:
    logits = forward(model, tokens[0:-1], nrecur, key)
    return cross_entropy_loss(tokens[1:], logits, mask[1:])

@partial(jax.jit, static_argnames=("forward", "nrecur"))
def batched_autoregressive_loss(model: Any, forward: Callable[[Any, jax.Array, int], jax.Array], tokens: jax.Array, mask: jax.Array, nrecur: int, key: jax.Array) -> jax.Array:
    batched_loss = jax.vmap(autoregressive_loss, (None, None, 0, 0, None, None), 0)(model, forward, tokens, mask, nrecur, key)
    return jnp.mean(batched_loss)

@partial(jax.jit, static_argnames=("forward", "nrecur"))
def loss(model: Any, forward: Callable[[Any, jax.Array, int], jax.Array], tokens_in: jax.Array, tokens_out: jax.Array, mask: jax.Array, nrecur: int, key: jax.Array) -> jax.Array:
    logits = forward(model, tokens_in, nrecur, key)
    return cross_entropy_loss(tokens_out, logits, mask)

@partial(jax.jit, static_argnames=("forward", "nrecur"))
def batched_loss(model: Any, forward: Callable[[Any, jax.Array, int], jax.Array], tokens_in: jax.Array, tokens_out: jax.Array, mask: jax.Array, nrecur: int, key: jax.Array) -> jax.Array:
    batched_loss = jax.vmap(loss, (None, None, 0, 0, 0, None, None), 0)(model, forward, tokens_in, tokens_out, mask, nrecur, key)
    return jnp.mean(batched_loss)

@partial(jax.jit, static_argnames=("forward", "nrecur"), donate_argnums=(0, 4))
def train_step_autoregressive(model: Any, forward: Callable[[Any, jax.Array, int], jax.Array], tokens: jax.Array, mask: jax.Array, opt: AdamW, nrecur: int, key: jax.Array) -> Tuple[jax.Array, Any, AdamW]:
    loss, grad = jax.value_and_grad(batched_autoregressive_loss)(model, forward, tokens, mask, nrecur, key)
    model, opt = update(model, grad, opt)
    return loss, model, opt

@partial(jax.jit, static_argnames=("forward", "nrecur"), donate_argnums=(0, 5))
def train_step(model: Any, forward: Callable[[Any, jax.Array, int, jax.Array], jax.Array], tokens_in: jax.Array, tokens_out: jax.Array, mask: jax.Array, opt: AdamW, nrecur: int, key: jax.Array) -> Tuple[jax.Array, Any, AdamW]:
    loss, grad = jax.value_and_grad(batched_loss)(model, forward, tokens_in, tokens_out, mask, nrecur, key)
    model, opt = update(model, grad, opt)
    return loss, model, opt