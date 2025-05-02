import json
import random
from dataclasses import dataclass
from typing import List, Optional, Callable, Iterator

import jax
import jax.numpy as jnp


HOLE_TOKEN = jnp.uint8(46)  # '.' in ASCII
NULL_TOKEN = jnp.uint8(95)  # '_' in ASCII, for subsequent layers

#
# 1. Utility: Loading puzzles & uniform spaces
#

def load_puzzles_from_json(filename: str):
    """
    Loads an array of puzzles from a JSON file.
    Each puzzle has:
      {
        "complete": List[List[int]],
        "partial":  List[List[Optional[int]]]
      }
    """
    with open(filename, "r") as f:
        puzzles = json.load(f)
    return puzzles  # list of dicts


def uniform_grid(x: jnp.uint8, size: int) -> List[List[jnp.uint8]]:
    """
    Returns a list-of-lists (size x size) filled with token x.
    """
    return [[x for _ in range(size)] for _ in range(size)]


def uniform_space(x: jnp.uint8, size: int, depth: int) -> List[List[List[jnp.uint8]]]:
    """
    Returns a list of grids, each of size x size, repeated depth times.
    """
    return [uniform_grid(x, size) for _ in range(depth)]


#
# 2. Interpolating the puzzle in the first layer
#

def alpha_interpolate(
    complete_grid: List[List[int]],
    partial_grid: List[List[Optional[int]]],
    alpha: float,
    rng: random.Random,
) -> List[List[Optional[int]]]:
    """
    For each cell:
      - with probability alpha, use complete_grid
      - with probability (1-alpha), use partial_grid
    """
    size = len(complete_grid)
    mixed_grid = []
    for r in range(size):
        row = []
        for c in range(size):
            if rng.random() < alpha:
                # Take from complete
                val = complete_grid[r][c]
            else:
                # Take from partial (might be None)
                val = partial_grid[r][c]
            row.append(val)
        mixed_grid.append(row)
    return mixed_grid


def to_token_grid(grid: List[List[Optional[int]]]) -> List[List[jnp.uint8]]:
    """
    Convert a 2D grid of optional ints -> tokens:
      - int n => ASCII code '0'+n  (1->'1'(49), 2->'2'(50), etc.)
      - None  => HOLE_TOKEN (46)
    """
    tokenized = []
    for row in grid:
        token_row = []
        for cell in row:
            if cell is None:
                token_row.append(HOLE_TOKEN)
            else:
                # For standard Sudoku 1..9 => '1'..'9'
                # If cell=1 => ASCII '1' = 49
                token_row.append(jnp.uint8(ord('0') + cell))
        tokenized.append(token_row)
    return tokenized


#
# 3. Data structures: Batch, JaxBatch
#

@dataclass
class Batch:
    """
    Each item is a "Space" with shape [depth, size, size].
    So `input[i]` is one puzzle's input of shape [depth][size][size].
    """
    input: List[List[List[List[jnp.uint8]]]]   # [batch_size][depth][size][size]
    output: List[List[List[List[jnp.uint8]]]]  # [batch_size][depth][size][size]
    mask: List[List[List[List[bool]]]]         # [batch_size][depth][size][size]


@dataclass
class JaxBatch:
    """
    Arrays of shape (batch_size, depth, size, size).
      - input, output: uint8
      - mask: bool
    """
    input: jnp.ndarray
    output: jnp.ndarray
    mask: jnp.ndarray

    @staticmethod
    def from_batch(b: Batch) -> "JaxBatch":
        inp = jnp.array(b.input, dtype=jnp.uint8)    # (B, D, S, S)
        out = jnp.array(b.output, dtype=jnp.uint8)   # (B, D, S, S)
        msk = jnp.array(b.mask, dtype=jnp.bool_)     # (B, D, S, S)
        return JaxBatch(input=inp, output=out, mask=msk)


#
# 4. Batcher with Depth & Null Tokens
#

class Batcher(Iterator[JaxBatch]):
    """
    A Batcher that:
      - Has puzzles loaded from a JSON file.
      - On each iteration, picks `batch_size` puzzles (with replacement).
      - alpha_func(iter) in [0,1] controls interpolation in the first layer.
      - The first layer is the puzzle, the rest of depth-1 layers are NULL_TOKEN.
      - The output's first layer is the full solution, the rest also NULL_TOKEN.
      - The mask's first layer is True where puzzle != HOLE_TOKEN, else False;
        subsequent layers are all False.
    """
    def __init__(
        self,
        puzzle_file: str,
        batch_size_func: Callable[[int], int],
        depth_func: Callable[[int], int],
        alpha_func: Callable[[int], float],
        seed: int,
    ):
        with open(puzzle_file, 'r') as f:
            self.puzzles = json.load(f)
        self.batch_size_func = batch_size_func
        self.depth_func = depth_func
        self.alpha_func = alpha_func
        self.iteration = 0
        self.rng = random.Random(seed)

    def __iter__(self):
        return self

    def __next__(self) -> JaxBatch:
        if not self.puzzles:
            raise StopIteration("No puzzles loaded.")

        alpha = self.alpha_func(self.iteration)
        alpha = max(0.0, min(1.0, alpha))  # clamp
        batch_size = self.batch_size_func(self.iteration)
        depth = self.depth_func(self.iteration)
        
        batch_input = []
        batch_output = []
        batch_mask = []

        for _ in range(batch_size):
            puzzle = self.rng.choice(self.puzzles)
            complete = puzzle["complete"]  # List[List[int]]
            partial = puzzle["partial"]    # List[List[Optional[int]]]

            # 1) Interpolate partial vs complete -> "mixed" puzzle
            mixed = alpha_interpolate(complete, partial, alpha, self.rng)

            # 2) Convert to tokens
            token_input_first_layer = to_token_grid(mixed)
            token_output_first_layer = to_token_grid(complete)

            # 3) Build rest of layers = uniform_space(NULL_TOKEN)
            size = len(token_input_first_layer)
            rest_input = uniform_space(NULL_TOKEN, size, depth - 1)
            rest_output = uniform_space(NULL_TOKEN, size, depth - 1)

            # 5) Combine layers
            space_input = [token_input_first_layer] + rest_input  # depth layers
            space_output = [token_output_first_layer] + rest_output
            space_mask = [[[True] * len(row) for row in token_input_first_layer]] + [
                [[False]*size for _ in range(size)]
                for _ in range(depth - 1)
            ]

            batch_input.append(space_input)
            batch_output.append(space_output)
            batch_mask.append(space_mask)

        # Convert to JaxBatch
        raw = Batch(
            input=batch_input,
            output=batch_output,
            mask=batch_mask,
        )
        jbatch = JaxBatch.from_batch(raw)
        self.iteration += 1
        return jbatch


#
# Example usage
#

if __name__ == "__main__":
    filename = "puzzles.json"

    # Example schedule functions
    def batch_size_schedule(it: int) -> int:
        return min(32, 2 * (it + 1))  # Increase from 2 to 32
    
    def depth_schedule(it: int) -> int:
        return min(5, 2 + it // 10)   # Increase from 2 to 5
        
    def alpha_schedule(it: int) -> float:
        return min(1.0, it / 10.0)    # Linear ramp from 0 to 1

    #key = jax.random.PRNGKey(42)
    batcher = Batcher(
        puzzle_file=filename,
        batch_size_func=batch_size_schedule,
        depth_func=depth_schedule,
        alpha_func=alpha_schedule,
        seed=42,
    )

    # Example: get 2 batches
    for _ in range(2):
        jbatch = next(batcher)
        iter_i = batcher.iteration - 1
        batch_size = batch_size_schedule(iter_i)
        depth = depth_schedule(iter_i)
        alpha = alpha_schedule(iter_i)
        print(f"\n--- BATCH {iter_i} ---")
        print(f"Parameters: batch_size={batch_size}, depth={depth}, alpha={alpha:.2f}")
        print("Shapes:")
        print(" input =", jbatch.input.shape)   # (batch_size, depth, size, size)
        print(" output=", jbatch.output.shape)  # (batch_size, depth, size, size)
        print(" mask  =", jbatch.mask.shape)

        # Let's peek at the first puzzle in the batch
        inp0 = jbatch.input[0]    # shape (depth, size, size)
        out0 = jbatch.output[0]   # shape (depth, size, size)
        msk0 = jbatch.mask[0]     # shape (depth, size, size)

        # We'll print the first layer input vs output as strings
        first_layer_in = inp0[0]
        first_layer_out = out0[0]
        size = first_layer_in.shape[0]

        def row_to_str(row_tokens):
            return "".join(chr(int(x)) for x in row_tokens)

        print("\nFirst puzzle, first layer [input]:")
        for r in range(size):
            print(row_to_str(first_layer_in[r]))

        print("\nFirst puzzle, first layer [output]:")
        for r in range(size):
            print(row_to_str(first_layer_out[r]))