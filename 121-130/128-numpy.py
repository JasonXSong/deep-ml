"""
Implement the Dynamic Tanh (DyT) function, a normalization-free transformation inspired by the Tanh function. DyT replaces layer normalization in Transformer architectures while preserving squashing behavior and enabling stable training.
Example:
Input:
x = np.array([[[0.14115588, 0.00372817, 0.24126647, 0.22183601]]])
gamma = np.ones((4,))
beta = np.zeros((4,))
alpha = 0.5
print(dynamic_tanh(x, alpha, gamma, beta))
Output:
[[[0.0705, 0.0019, 0.1201, 0.1105]]]
Reasoning:
Each element in the input is scaled by alpha, passed through tanh, and then scaled by gamma and shifted by beta. This mimics the squashing behavior of layer normalization without explicitly using statistics.
"""


import numpy as np

def dynamic_tanh(x: np.ndarray, alpha: float, gamma: float, beta: float) -> list[float]:
    # Your code here
    return np.tanh(alpha * x) * gamma + beta


if __name__ == "__main__":
    x = np.array([[[0.14115588, 0.00372817, 0.24126647, 0.22183601]]])
    gamma = np.ones((4,))
    beta = np.zeros((4,))
    alpha = 0.5
    print(dynamic_tanh(x, alpha, gamma, beta))
