"""
Implement masked self-attention, a variation of the attention mechanism used in sequence modeling tasks such as text generation. Your task is to compute masked self-attention using query (Q), key (K), value (V) matrices and an attention mask.

Example:
Input:
masked_attention(Q, K, V, mask)
Output:
[[547. 490. 399. 495. 485. 439. 645. 393.]
 [547. 490. 399. 495. 485. 439. 645. 393.]
 [471. 472. 429. 538. 377. 450. 531. 362.]
 [471. 472. 429. 538. 377. 450. 531. 362.]
 [471. 472. 429. 538. 377. 450. 531. 362.]
 [471. 472. 429. 538. 377. 450. 531. 362.]]
Reasoning:
The function computes self-attention by applying a mask to restrict information flow, ensuring causal dependencies are maintained.
"""


import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute masked self-attention.
    """
    # Your code here
    QK = np.matmul(Q, K.T)
    d_k = np.sqrt(K.shape[-1])
    QK = QK / d_k
    QK_masked = QK + mask
    QK_weights = np.exp(QK_masked - np.max(QK_masked, axis=-1, keepdims=True))
    QK_softmax = QK_weights / np.sum(QK_weights, axis=-1, keepdims=True)
    QKV = np.matmul(QK_softmax, V)
    return QKV


if __name__ == "__main__":
    np.random.seed(42)
    X = np.arange(48).reshape(6,8)
    X = np.random.permutation(X.flatten()).reshape(6, 8)
    mask = np.triu(np.ones((6, 6))*(-np.inf), k=1)
    W_q = np.random.randint(0,4,size=(8,8))
    W_k = np.random.randint(0,5,size=(8,8))
    W_v = np.random.randint(0,6,size=(8,8))
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    print(masked_attention(Q, K, V, mask))
