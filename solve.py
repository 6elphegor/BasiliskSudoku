import jax
import jax.numpy as jnp
import jax.random as random

import sys
import time

from typing import Optional, Any
import pickle

import itertools
from itertools import islice
import copy
from functools import partial

import model
from model.token_image_transformer import TokenImageTransformer, random_token_image_transformer, token_image_transformer_forward_embedding, token_image_transformer_forward_intermediate, token_image_transformer_forward_output

import sudoku
from sudoku.sudoku import Batcher, HOLE_TOKEN, NULL_TOKEN


def load_model(file_path: str) -> Optional[Any]:
    try:
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model
    except (FileNotFoundError, pickle.PickleError):
        return None

key = random.key(0)
key, train_key = random.split(key, num=2)

print("JAX device: ", jax.devices())

# 9x9 sudoku puzzles
width = 9
depth = 1
nblocks = 15

# Load model
model = load_model("model.pkl")
if model is None:
    print("Initializing new model")
    key, key_model = random.split(key)
    model = random_token_image_transformer(key=key_model, nvocab=256, nstate=256, n_q_head=8, n_kv_head=8, mult=4, width=width, depth=depth, n_blocks=nblocks)
else:
    print("Model successfully loaded")

# Original puzzles should store direct numbers, with 0 for holes

def print_sudoku(puzzle: Any) -> None:
    """Pretty prints a Sudoku puzzle with grid lines."""
    horizontal_line = "+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+"
    for i in range(9):
        if i % 3 == 0:
            print(horizontal_line)
        for j in range(9):
            if j % 3 == 0:
                print("|", end=" ")
            value = puzzle[i][j]
            if value == 0:
                print(".", end=" ")
            else:
                print(int(value), end=" ")
            if j == 8:
                print("|")
    print(horizontal_line)

def convert_to_ascii(puzzle: Any, num_depths: int) -> Any:
    """Convert direct numbers (0-9) to ASCII representation and create depth dimension."""
    ascii_nums = jnp.array([ord(str(i)) for i in range(1, 10)])
    lookup = jnp.concatenate([jnp.array([HOLE_TOKEN]), ascii_nums])
    first_depth = lookup[puzzle]
    
    null_layers = jnp.full((num_depths - 1, *puzzle.shape), NULL_TOKEN)
    return jnp.concatenate([first_depth[None, ...], null_layers])

# Initialize puzzles
puzzle = jnp.array([
    [9, 8, 7, 4, 1, 2, 3, 5, 6],
    [2, 6, 5, 8, 3, 9, 7, 4, 1],
    [3, 4, 1, 5, 7, 6, 8, 2, 9],
    [1, 2, 6, 9, 5, 8, 4, 7, 3],
    [5, 3, 4, 7, 6, 1, 2, 9, 8],
    [7, 9, 8, 3, 2, 4, 6, 1, 5],
    [8, 1, 9, 6, 4, 7, 5, 3, 2],
    [4, 5, 2, 1, 8, 3, 9, 6, 7],
    [6, 7, 3, 2, 9, 5, 1, 8, 4]
])

holed_puzzle = jnp.array([
    [0, 8, 7, 0, 0, 0, 3, 5, 0],
    [0, 6, 0, 0, 3, 9, 0, 0, 0],
    [0, 0, 1, 5, 0, 6, 8, 0, 0],
    [0, 0, 6, 9, 5, 0, 4, 7, 0],
    [0, 3, 0, 0, 6, 0, 0, 0, 0],
    [7, 9, 8, 3, 0, 0, 0, 0, 0],
    [0, 1, 9, 6, 0, 0, 0, 0, 0],
    [4, 0, 2, 0, 0, 0, 0, 6, 7],
    [0, 0, 0, 2, 0, 0, 0, 8, 0]
])

# Print initial state
print("Initial holed puzzle:")
print_sudoku(holed_puzzle)

# Position cursor for updates (save space for the puzzle display)
print("\n" * 13)  # Add space for puzzle display
#sys.stdout.write("\033[13A")  # Move cursor up 13 lines
sys.stdout.flush()

# Convert to ASCII and add depth dimension for network
puzzle_ascii = convert_to_ascii(puzzle, depth)
holed_puzzle_ascii = convert_to_ascii(holed_puzzle, depth)

key, key_latent = random.split(key)

# Get embedding representation
embedding = token_image_transformer_forward_embedding(model, holed_puzzle_ascii)
latent = token_image_transformer_forward_intermediate(model, embedding, embedding, 1, key_latent)

# Run iterations with smooth updates
n_its = 100
for j in range(n_its):
    output = token_image_transformer_forward_output(model, latent)
    
    # Get predicted numbers from logits and remove depth dimension
    ascii_preds = jnp.argmax(output, axis=-1)[0]
    
    # Convert back using lookup table
    lookup = jnp.zeros(256, dtype=jnp.int32)
    for i in range(1, 10):
        lookup = lookup.at[ord(str(i))].set(i)
    lookup = lookup.at[HOLE_TOKEN].set(0)
    predictions = lookup[ascii_preds]
    
    # Clear previous puzzle display (move up 13 lines)
    sys.stdout.write("\033[13A")
    
    # Print current state
    print_sudoku(predictions)
    sys.stdout.flush()
    time.sleep(0.1)  # Small delay for visualization

    if (predictions == puzzle).all():
        print(f"\nSolution found after {j + 1} iterations.")
        exit()
    
    key, key_latent = random.split(key)
    latent = token_image_transformer_forward_intermediate(model, embedding, latent, 1, key_latent)

print("\nNo solution found")