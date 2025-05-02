# 🐍 BasiliskSudoku

BasiliskSudoku is a deep learning model designed to solve 9x9 Sudoku puzzles. It uses a custom transformer architecture with recurrent latent space reasoning.

## Overview

BasiliskSudoku leverages the power of JAX for efficient GPU training of a specialized transformer model. The model processes Sudoku boards and learns to reason about the game constraints through a recurrent latent space architecture. The training process consists of two stages:
1. Initial training with fewer recurrences for faster convergence
2. Slower training phase with increased recurrences for improved accuracy

## Installation & Setup

### Prerequisites
- CUDA-enabled GPU
- Python 3.6+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/6elphegor/BasiliskSudoku.git
cd BasiliskSudoku
```

2. Download the training dataset:
```bash
wget -O sudoku/puzzles.json.gz https://huggingface.co/datasets/6elphegor/Sudoku/resolve/main/puzzles.json.gz?download=true
gunzip -f sudoku/puzzles.json.gz
```

3. Create and activate a virtual environment:
```bash
python3 -m venv myenv
source myenv/bin/activate
```

4. Install JAX with CUDA support:
```bash
pip install -U "jax[cuda12]"
```

## Training

The training process involves two stages:

### Stage 1: Initial Training
```bash
python3 main.py
```

Train the model for approximately 13,000 iterations. The initial training uses fewer recurrences (`nrecur = 5`) to allow for faster learning.

### Stage 2: Fine-tuning
After the initial training completes, modify the training script for stage 2:
```bash
sed -i 's/^firstStage = True/firstStage = False/' main.py
```

Resume training:
```bash
python3 main.py
```

The second stage uses more recurrences (`nrecur = 15`) and a reduced learning rate for better accuracy.

## Performance

**Stage 1**
```
At epoch 130 step 13097 train loss is 0.1245272234082222 nrecur = 5, alpha = 0.0, depth = 1 dtype is float32
At epoch 130 step 13098 train loss is 0.5645176768302917 nrecur = 5, alpha = 0.0, depth = 1 dtype is float32
At epoch 130 step 13099 train loss is 0.260018527507782 nrecur = 5, alpha = 0.0, depth = 1 dtype is float32
```

**Stage 2**
```
At epoch 60 step 6098 train loss is 0.01949063129723072 nrecur = 15, alpha = 0.0, depth = 1 dtype is float32
At epoch 60 step 6099 train loss is 0.006419305689632893 nrecur = 15, alpha = 0.0, depth = 1 dtype is float32
```

Note that the step count resets.

## Testing

To solve a Sudoku puzzle using the trained model:
```bash
python3 solve.py
```

### Example Output:
```
Initial holed puzzle:
+-------+-------+-------+
| . 8 7 | . . . | 3 5 . |
| . 6 . | . 3 9 | . . . |
| . . 1 | 5 . 6 | 8 . . |
+-------+-------+-------+
| . . 6 | 9 5 . | 4 7 . |
| . 3 . | . 6 . | . . . |
| 7 9 8 | 3 . . | . . . |
+-------+-------+-------+
| . 1 9 | 6 . . | . . . |
| 4 . 2 | . . . | . 6 7 |
| . . . | 2 . . | . 8 . |
+-------+-------+-------+

+-------+-------+-------+
| 9 8 7 | 4 1 2 | 3 5 6 |
| 2 6 5 | 8 3 9 | 7 4 1 |
| 3 4 1 | 5 7 6 | 8 2 9 |
+-------+-------+-------+
| 1 2 6 | 9 5 8 | 4 7 3 |
| 5 3 4 | 7 6 1 | 2 9 8 |
| 7 9 8 | 3 2 4 | 6 1 5 |
+-------+-------+-------+
| 8 1 9 | 6 4 7 | 5 3 2 |
| 4 5 2 | 1 8 3 | 9 6 7 |
| 6 7 3 | 2 9 5 | 1 8 4 |
+-------+-------+-------+

Solution found after 8 iterations.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

If you have any questions or feedback, please open an issue on GitHub.
