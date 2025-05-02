import jax
import jax.numpy as jnp
import jax.random as random
import jax.tree_util as jtu

from typing import Optional, Any
import pickle

import itertools
from itertools import islice
import copy
from functools import partial

import model
from model.token_image_transformer import TokenImageTransformer, random_token_image_transformer, token_image_transformer_forward
from train import train_step
from optimizer import AdamW, new_adamw

import sudoku
from sudoku.sudoku import Batcher

def save_model(model: Any, file_path: str):
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

def load_model(file_path: str) -> Optional[Any]:
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model
    except (FileNotFoundError, pickle.PickleError):
        return None
    

def to_bf16_if_float(x):
    if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
        return x.astype(jnp.bfloat16)
    return x
    
@partial(jax.jit, static_argnames=("nrecur"))
def forward(model: TokenImageTransformer, tokens: jax.Array, nrecur: int, key: jax.Array) -> jax.Array:
    return token_image_transformer_forward(model, tokens, nrecur, key)

key = random.key(369)
key, train_key = random.split(key)

print("JAX device: ", jax.devices())

# Load dataset
epoch_size = 100
num_epochs = 100000

# 9x9 sudoku puzzles
width = 9
depth = 1
nblocks = 10

firstStage = True

if firstStage:
    nrecur = 5
else:
    nrecur = 15

def batch_size_schedule(it: int) -> int:
    return 1

# 1st layer is sudoku puzzle, others are hidden reasoning nodes
def depth_schedule(it: int) -> int:
    return depth

# alpha is the probability of sampling the solution number for each hole
def alpha_schedule(it: int) -> float:
    #start = 0.9
    #return start - min(start, it / 10000.0)
    return 0.0

# reasoning depth
def nrecur_schedule(it: int) -> int:
    #return 2 + min(it // 100, 8)
    return nrecur

batcher = Batcher(
    puzzle_file="sudoku/puzzles.json",
    batch_size_func=batch_size_schedule,
    depth_func=depth_schedule,
    alpha_func=alpha_schedule,
    seed=42,
)

# Load model
model = load_model("model.pkl")
if model is None:
    print("Initializing new model")
    key, key_model = random.split(key)
    model = random_token_image_transformer(key=key_model, nvocab=256, nstate=256, n_q_head=8, n_kv_head=8, mult=4, width=width, depth=depth, n_blocks=nblocks)
else:
    print("Model successfully loaded")

# Load optimizer
opt = load_model("opt.pkl")
if opt is None:
    print("Initializing new optimizer")
    opt = new_adamw(lr=3e-4)
else:
    print("Optimizer successfully loaded")

# Slow down for second stage
if not firstStage:
    opt = opt._replace(lr=3e-5)

# convert to bf16
#model = jtu.tree_map(lambda x: x.astype(jnp.bfloat16), model)
#opt = jtu.tree_map(to_bf16_if_float, opt)

# Training loop
losses_train = []
i = 0

for epoch_idx in range(num_epochs):
    # if i > 20000:
    #     break
    for batch_train in islice(batcher, epoch_size):
        input = batch_train.input
        output = batch_train.output
        mask = batch_train.mask

        batch_size = batch_size_schedule(i)
        alpha = alpha_schedule(i)
        nrecur = nrecur_schedule(i)

        key, key_train = random.split(key)
        loss_train, model, opt = train_step(model, forward, input, output, mask, opt, nrecur, key_train)

        print(f'dtype is {loss_train.dtype}')

        losses_train.append(loss_train.item())

        print(f'At epoch {epoch_idx} step {i} train loss is {loss_train}')
        print(f'nrecur = {nrecur}, alpha = {alpha}, depth = {depth}')

        i += 1

    if epoch_idx % 10 == 0:
        print("saving losses")
        with open('losses_train.txt', "w") as file:
            file.write("\n".join(map(str, losses_train)))

        print("Saving model")
        save_model(model, "model.pkl")
        save_model(opt, "opt.pkl")

