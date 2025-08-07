"""
Implement a Python function that applies the GELU (Gaussian Error Linear Unit) activation function to a NumPy array of logits. Round each output to four decimal places and return the result as a NumPy array of the same shape.

Example:
Input:
np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
Output:
[-0.0454 -0.1588 0. 0.8412 1.9546]
Reasoning:
Each value in the input array is passed through the GELU activation function using the approximation formula. For example, GELU(1.0) ≈ 0.841192, which rounds to 0.8412. Similarly, GELU(-2.0) ≈ -0.045402, which rounds to -0.0454. Since we're using numpy, it uses vectorized operations and applies the formula to each element in the array. Then, we rounded it to four decimal places to produce the output array.
"""


import numpy as np


def GeLU(x: np.ndarray) -> np.ndarray:
    # Your code here
    scores = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * (x ** 3))))
    return np.round(scores, 4)


if __name__ == "__main__":
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(GeLU(x))
