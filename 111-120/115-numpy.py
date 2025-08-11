"""
Implement a function that performs Batch Normalization on a 4D NumPy array representing a batch of feature maps in the BCHW format (batch, channels, height, width). The function should normalize the input across the batch and spatial dimensions for each channel, then apply scale (gamma) and shift (beta) parameters. Use the provided epsilon value to ensure numerical stability.

Example:
Input:
B, C, H, W = 2, 2, 2, 2; np.random.seed(42); X = np.random.randn(B, C, H, W); gamma = np.ones(C).reshape(1, C, 1, 1); beta = np.zeros(C).reshape(1, C, 1, 1)
Output:
[[[[ 0.42859934, -0.51776438], [ 0.65360963,  1.95820707]], [[ 0.02353721,  0.02355215], [ 1.67355207,  0.93490043]]], [[[-1.01139563,  0.49692747], [-1.00236882, -1.00581468]], [[ 0.45676349, -1.50433085], [-1.33293647, -0.27503802]]]]
Reasoning:
The input X is a 2x2x2x2 array. For each channel, compute the mean and variance across the batch (B), height (H), and width (W) dimensions. Normalize X using (X - mean) / sqrt(variance + epsilon), then scale by gamma and shift by beta. The output matches the expected normalized values.
"""


import numpy as np

def batch_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    # Your code here
    mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
    variance = np.var(X, axis=(0, 2, 3), keepdims=True)
    X_normalized = (X - mean) / np.sqrt(variance + epsilon)
    return X_normalized * gamma + beta


if __name__ == "__main__":
    B, C, H, W = 2, 2, 2, 2
    np.random.seed(42)
    X = np.random.randn(B, C, H, W)
    gamma = np.ones(C).reshape(1, C, 1, 1)
    beta = np.zeros(C).reshape(1, C, 1, 1)
    print(batch_normalization(X, gamma, beta))
