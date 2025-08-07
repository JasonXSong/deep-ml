"""
Implement a function that creates a simple residual block using NumPy. The block should take a 1D input array, process it through two weight layers (using matrix multiplication), apply ReLU activations, and add the original input via a shortcut connection before a final ReLU activation.

Example:
Input:
x = np.array([1.0, 2.0]), w1 = np.array([[1.0, 0.0], [0.0, 1.0]]), w2 = np.array([[0.5, 0.0], [0.0, 0.5]])
Output:
[1.5, 3.0]
Reasoning:
The input x is [1.0, 2.0]. First, compute w1 @ x = [1.0, 2.0], apply ReLU to get [1.0, 2.0]. Then, compute w2 @ [1.0, 2.0] = [0.5, 1.0]. Add the shortcut x to get [0.5 + 1.0, 1.0 + 2.0] = [1.5, 3.0]. Final ReLU gives [1.5, 3.0].
"""


import numpy as np


def relu(x):
    return np.maximum(x, 0)

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    # Your code here
    a1 = np.matmul(w1, x)
    z1 = relu(a1)
    a2 = np.matmul(w2, z1)
    z2 = relu(a2 + x)
    return z2


if __name__ == "__main__":
	x = np.array([1.0, 2.0])
	w1 = np.array([[1.0, 0.0], [0.0, 1.0]])
	w2 = np.array([[0.5, 0.0], [0.0, 0.5]])
	print(residual_block(x, w1, w2))