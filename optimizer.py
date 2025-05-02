import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Any
from jax.tree_util import tree_map

class AdamW(NamedTuple):
    lr: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    moment_first: Any
    moment_second: Any
    t: int

def new_adamw(lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 1e-2) -> AdamW:
    """Initializes a new AdamW optimizer state."""
    return AdamW(
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        moment_first=None,
        moment_second=None,
        t=0
    )

def init_moments(params) -> Tuple[Any, Any]:
    moment_first = tree_map(lambda p: jnp.zeros_like(p), params)
    moment_second = tree_map(lambda p: jnp.zeros_like(p), params)
    return moment_first, moment_second

@jax.jit
def update(params, grads, opt: AdamW) -> Tuple[Any, AdamW]:
    if opt.moment_first is None or opt.moment_second is None:
        moment_first, moment_second = init_moments(params)
    else:
        moment_first = opt.moment_first
        moment_second = opt.moment_second

    t = opt.t + 1
    
    moment_first = tree_map(lambda m, g: opt.beta1 * m + (1 - opt.beta1) * g, moment_first, grads)
    moment_second = tree_map(lambda v, g: opt.beta2 * v + (1 - opt.beta2) * (g ** 2), moment_second, grads)

    # Scale to counteract moment bias
    lr_t = opt.lr * jnp.sqrt(1 - opt.beta2 ** t) / (1 - opt.beta1 ** t)

    params = tree_map(
        lambda p, m, v: p - lr_t * (m / (jnp.sqrt(v) + opt.eps)) - opt.lr * opt.weight_decay * p,
        params, moment_first, moment_second
    )

    new_opt = AdamW(
        lr=opt.lr,
        beta1=opt.beta1,
        beta2=opt.beta2,
        eps=opt.eps,
        weight_decay=opt.weight_decay,
        moment_first=moment_first,
        moment_second=moment_second,
        t=t
    )

    return params, new_opt
