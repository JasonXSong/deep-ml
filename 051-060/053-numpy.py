"""
Task: Implement the Self-Attention Mechanism
Your task is to implement the self-attention mechanism, which is a fundamental component of transformer models, widely used in natural language processing and computer vision tasks. The self-attention mechanism allows a model to dynamically focus on different parts of the input sequence when generating a contextualized representation.

Your function should return the self-attention output as a numpy array.

Example:
Input:
import numpy as np

X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)

print(output)
Output:
# [[1.660477 2.660477]
#  [2.339523 3.339523]]
Reasoning:
The self-attention mechanism calculates the attention scores for each input, determining how much focus to put on other inputs when generating a contextualized representation. The output is the weighted sum of the values based on the attention scores.
"""


import numpy as np

def self_attention(Q, K, V):
    QK = np.matmul(Q, K.T) / np.sqrt(K.shape[-1])
    QK_weights = np.exp(QK - np.max(QK, axis=-1, keepdims=True))
    QK_softmax = QK_weights / np.sum(QK_weights, axis=-1, keepdims=True)

    return np.matmul(QK_softmax, V)


def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return X @ W_q, X @ W_k, X @ W_v


if __name__ == "__main__":
    X = np.array([[1, 0], [0, 1]])
    W_q = np.array([[1, 0], [0, 1]])
    W_k = np.array([[1, 0], [0, 1]])
    W_v = np.array([[1, 2], [3, 4]])
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    print(self_attention(Q, K, V))

    X = np.array([[1, 1], [1, 0]])
    W_q = np.array([[1, 0], [0, 1]])
    W_k = np.array([[1, 0], [0, 1]])
    W_v = np.array([[1, 2], [3, 4]])
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    output = self_attention(Q, K, V)
    print(output)
