"""
Implement the Swish activation function, a self-gated activation function that has shown superior performance in deep neural networks compared to ReLU. Your task is to compute the Swish value for a given input.

Example:
Input:
swish(1)
Output:
0.7311
Reasoning:
For x = 1, the Swish activation is calculated as Swish(x) = x * sigmoid(x), where sigmoid(x) = 1 / (1 + e^{-x}). Substituting the value, Swish(1) = 1 * 1 / (1+e^{-1}) =0.7311.
"""


import numpy as np


def swish(x: float) -> float:
    """
    Implements the Swish activation function.

    Args:
        x: Input value

    Returns:
        The Swish activation value
    """
    # Your code here
    sigmoid = 1 / (1 + np.exp(-x))
    return round(x * sigmoid, 4)


if __name__ == "__main__":
	print(swish(1))
